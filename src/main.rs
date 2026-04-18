//! # Physarum Toy
//!
//! A GPU-accelerated simulation of Physarum polycephalum (slime mold) using Bevy and WGSL compute shaders.
//! This simulation implements the agent-based model described by Jeff Jones, where simple agents
//! follow pheromone trails and deposit their own trails, leading to complex emergent patterns.

use bevy::dev_tools::fps_overlay::FrameTimeGraphConfig;
use bevy::{
    dev_tools::fps_overlay::{FpsOverlayConfig, FpsOverlayPlugin},
    prelude::*,
    render::{
        Render, RenderApp,
        extract_resource::{ExtractResource, ExtractResourcePlugin},
        render_asset::RenderAssets,
        render_graph::{self, RenderGraph, RenderLabel},
        render_resource::*,
        renderer::{RenderContext, RenderDevice, RenderQueue},
        storage::{GpuShaderStorageBuffer, ShaderStorageBuffer},
        texture::GpuImage,
    },
};
use bevy::window::WindowResolution;
use bevy_egui::{egui, EguiContexts, EguiPlugin, EguiPrimaryContextPass};
use bytemuck::{Pod, Zeroable};
use rand::RngExt;

/// The total number of agents in the simulation.
const AGENT_COUNT: u32 = 1_000_000;
/// The resolution of the simulation and window.
const SIZE: (u32, u32) = (1920, 1080);

const DEFAULT_SENSOR_ANGLE: f32 = 20.0f32.to_radians();
const DEFAULT_SENSOR_DIST: f32 = 15.0;
const DEFAULT_TURN_SPEED: f32 = 550.0f32.to_radians();
const DEFAULT_MOVE_SPEED: f32 = 50.0;
const DEFAULT_DECAY: f32 = 1.0;
const DEFAULT_DIFFUSE_SPEED: f32 = 60.0;

fn main() {
    App::new()
        .add_plugins(DefaultPlugins.set(WindowPlugin {
            primary_window: Some(Window {
                title: "Physarum Toy".to_string(),
                resolution: WindowResolution::new(SIZE.0, SIZE.1),
                ..default()
            }),
            ..default()
        }))
        .add_plugins(EguiPlugin::default())
        .add_plugins(FpsOverlayPlugin {
            config: FpsOverlayConfig {
                text_config: TextFont {
                    font_size: 10.0,
                    ..default()
                },
                text_color: Color::srgb(1.0, 0.0, 0.0),
                frame_time_graph_config: FrameTimeGraphConfig {
                    enabled: false,
                    ..default()
                },
                enabled: true,
                ..default()
            },
        })
        .add_plugins(ExtractResourcePlugin::<PhysarumResources>::default())
        .add_plugins(ExtractResourcePlugin::<PhysarumConfigResource>::default())
        .insert_resource(ClearColor(Color::BLACK))
        .insert_resource(PhysarumConfigResource {
            config: PhysarumConfig {
                sensor_angle: DEFAULT_SENSOR_ANGLE,
                sensor_dist: DEFAULT_SENSOR_DIST,
                turn_speed: DEFAULT_TURN_SPEED,
                move_speed: DEFAULT_MOVE_SPEED,
                decay: DEFAULT_DECAY,
                width: SIZE.0,
                height: SIZE.1,
                delta_time: 0.0,
                diffuse_speed: DEFAULT_DIFFUSE_SPEED,
                _pad0: 0.0,
                _pad1: 0.0,
                _pad2: 0.0,
            },
        })
        .add_systems(Startup, setup)
        .add_systems(Update, (update_config, move_fps_overlay))
        .add_systems(EguiPrimaryContextPass, physarum_ui)
        .add_plugins(PhysarumComputePlugin)
        .run();
}

/// Resources required for the Physarum simulation on the GPU.
#[derive(Resource, Clone, ExtractResource)]
struct PhysarumResources {
    /// Buffer containing all agent agents.
    agents: Handle<ShaderStorageBuffer>,
    /// The trail map texture where pheromones are deposited and sensed.
    trail_map: Handle<Image>,
    /// The compute shader handle.
    shader: Handle<Shader>,
}

/// A wrapper resource for the simulation configuration, allowing it to be extracted to the render world.
#[derive(Resource, ExtractResource, Clone)]
struct PhysarumConfigResource {
    config: PhysarumConfig,
}

/// Representation of a single agent.
#[repr(C)]
#[derive(Copy, Clone, Debug, Pod, Zeroable, ShaderType)]
struct Agent {
    /// 2D position in the simulation space.
    pos: Vec2,
    /// Orientation angle in radians.
    angle: f32,
    /// Padding for GPU alignment.
    _pad: f32,
}

/// Configuration parameters for the Physarum simulation.
#[repr(C)]
#[derive(Copy, Clone, Debug, Pod, Zeroable, ShaderType)]
struct PhysarumConfig {
    /// Angle at which the sensors are offset from the agent's forward direction.
    sensor_angle: f32,
    /// Distance from the agent to its sensors.
    sensor_dist: f32,
    /// Speed at which the agent turns towards pheromones.
    turn_speed: f32,
    /// Speed at which the agent moves forward.
    move_speed: f32,
    /// Rate of pheromone decay over time.
    decay: f32,
    /// Width of the simulation area.
    width: u32,
    /// Height of the simulation area.
    height: u32,
    /// Time elapsed since the last frame.
    delta_time: f32,
    /// Speed of pheromone diffusion.
    diffuse_speed: f32,
    /// Padding for WebGPU 16-byte uniform alignment (size must be multiple of 16).
    _pad0: f32,
    /// Padding for WebGPU 16-byte uniform alignment (size must be multiple of 16).
    _pad1: f32,
    /// Padding for WebGPU 16-byte uniform alignment (size must be multiple of 16).
    _pad2: f32,
}

/// Initializes the simulation resources, agents, and camera.
fn setup(
    mut commands: Commands,
    mut images: ResMut<Assets<Image>>,
    mut buffers: ResMut<Assets<ShaderStorageBuffer>>,
    asset_server: Res<AssetServer>,
) {
    // Create trail map image
    let mut image = Image::new_fill(
        Extent3d {
            width: SIZE.0,
            height: SIZE.1,
            depth_or_array_layers: 1,
        },
        TextureDimension::D2,
        &[0, 0, 0, 0, 0, 0, 0, 60],
        TextureFormat::Rgba16Float,
        bevy::asset::RenderAssetUsages::RENDER_WORLD,
    );
    image.texture_descriptor.usage |=
        TextureUsages::STORAGE_BINDING | TextureUsages::TEXTURE_BINDING;
    let trail_map_handle = images.add(image);

    // Initialize agents
    let mut rng = rand::rng();
    let agents_data: Vec<Agent> = (0..AGENT_COUNT)
        .map(|_| {
            let angle = rng.random_range(0.0..std::f32::consts::TAU);
            Agent {
                pos: Vec2::new(
                    rng.random_range(0.0..SIZE.0 as f32),
                    rng.random_range(0.0..SIZE.1 as f32),
                ),
                angle,
                _pad: 0.0,
            }
        })
        .collect();
    let agents_buffer = buffers.add(ShaderStorageBuffer::from(agents_data));
    let shader = asset_server.load("shaders/physarum.wgsl");

    commands.insert_resource(PhysarumResources {
        agents: agents_buffer,
        trail_map: trail_map_handle.clone(),
        shader,
    });

    commands.spawn(Camera2d);

    // Display the trail map
    commands.spawn(Sprite {
        image: trail_map_handle,
        custom_size: Some(Vec2::new(SIZE.0 as f32, SIZE.1 as f32)),
        ..default()
    });
}

/// Updates the delta time in the simulation configuration.
fn update_config(time: Res<Time>, mut config_res: ResMut<PhysarumConfigResource>) {
    config_res.config.delta_time = time.delta_secs();
}

/// Adjusts the position of the FPS overlay to the bottom-left corner.
fn move_fps_overlay(mut query: Query<(&mut Node, &GlobalZIndex)>) {
    for (mut node, z_index) in &mut query {
        if z_index.0 == i32::MAX - 32 {
            node.top = Val::Auto;
            node.bottom = Val::Px(10.0);
            node.left = Val::Px(10.0);
        }
    }
}

/// System that draws the configuration UI.
fn physarum_ui(mut contexts: EguiContexts, mut config_res: ResMut<PhysarumConfigResource>) {
    let ctx = match contexts.ctx_mut() {
        Ok(ctx) => ctx,
        Err(_) => return,
    };

    egui::SidePanel::right("config_panel")
        .default_width(250.0)
        .show(ctx, |ui| {
            ui.heading("Physarum Config");
            ui.add_space(10.0);

            let config = &mut config_res.config;

            ui.label("Sensor Angle (deg)")
                .on_hover_text("The angle (in degrees) at which the left and right sensors are offset from the agent's forward direction.");
            let mut sensor_angle_deg = config.sensor_angle.to_degrees();
            if ui
                .add(egui::Slider::new(&mut sensor_angle_deg, 0.0..=180.0))
                .changed()
            {
                config.sensor_angle = sensor_angle_deg.to_radians();
            }

            ui.label("Sensor Distance")
                .on_hover_text("The distance (in pixels) from the agent to its sensors.");
            ui.add(egui::Slider::new(&mut config.sensor_dist, 0.0..=100.0));

            ui.label("Turn Speed (deg/s)")
                .on_hover_text("The speed at which the agent turns towards pheromones (in degrees per second).");
            let mut turn_speed_deg = config.turn_speed.to_degrees();
            if ui
                .add(egui::Slider::new(&mut turn_speed_deg, 0.0..=3600.0))
                .changed()
            {
                config.turn_speed = turn_speed_deg.to_radians();
            }

            ui.label("Move Speed")
                .on_hover_text("The speed at which the agent moves forward (in pixels per second).");
            ui.add(egui::Slider::new(&mut config.move_speed, 0.0..=500.0));

            ui.label("Evaporation Time (s)")
                .on_hover_text("The number of seconds it takes for a trail to fully evaporate (linear decay).");
            let mut evap_time = if config.decay > 0.0 { 1.0 / config.decay } else { 10.0 };
            if ui.add(egui::Slider::new(&mut evap_time, 0.1..=10.0)).changed() {
                config.decay = 1.0 / evap_time;
            }

            ui.add_space(20.0);
            if ui.button("Reset to Defaults").clicked() {
                config.sensor_angle = DEFAULT_SENSOR_ANGLE;
                config.sensor_dist = DEFAULT_SENSOR_DIST;
                config.turn_speed = DEFAULT_TURN_SPEED;
                config.move_speed = DEFAULT_MOVE_SPEED;
                config.decay = DEFAULT_DECAY;
            }
        });
}

// --- Compute Infrastructure ---

/// Plugin responsible for setting up the compute shader pipeline and render graph nodes.
struct PhysarumComputePlugin;

/// Label identifying the Physarum compute node in the render graph.
#[derive(RenderLabel, Debug, Hash, PartialEq, Eq, Clone)]
struct PhysarumLabel;

impl Plugin for PhysarumComputePlugin {
    fn build(&self, app: &mut App) {
        let render_app = app.sub_app_mut(RenderApp);
        render_app
            .init_resource::<PhysarumPipeline>()
            .add_systems(Render, prepare_bind_group);

        let mut render_graph = render_app.world_mut().resource_mut::<RenderGraph>();
        render_graph.add_node(PhysarumLabel, PhysarumNode);
        render_graph.add_node_edge(PhysarumLabel, bevy::render::graph::CameraDriverLabel);
    }
}

/// Holds the compute pipeline IDs and bind group layout for the simulation.
#[derive(Resource, Default)]
struct PhysarumPipeline {
    /// Pipeline for the agent simulation step.
    simulate_pipeline: Option<CachedComputePipelineId>,
    /// Pipeline for the pheromone diffusion and decay step.
    diffuse_pipeline: Option<CachedComputePipelineId>,
    /// The common bind group layout used by both pipelines.
    bind_group_layout: Option<BindGroupLayout>,
}

/// Wrapper for the GPU bind group used in the compute shader.
#[derive(Resource)]
struct PhysarumBindGroup(BindGroup);

/// Prepares the bind group and initializes pipelines for the compute shader.
fn prepare_bind_group(
    mut commands: Commands,
    mut pipeline: ResMut<PhysarumPipeline>,
    render_device: Res<RenderDevice>,
    render_queue: Res<RenderQueue>,
    render_assets: Res<RenderAssets<GpuImage>>,
    render_buffers: Res<RenderAssets<GpuShaderStorageBuffer>>,
    resources: Option<Res<PhysarumResources>>,
    config_res: Option<Res<PhysarumConfigResource>>,
    pipeline_cache: Res<PipelineCache>,
) {
    let (Some(resources), Some(config_res)) = (resources, config_res) else {
        return;
    };

    if pipeline.bind_group_layout.is_none() {
        let entries = vec![
            BindGroupLayoutEntry {
                binding: 0,
                visibility: ShaderStages::COMPUTE,
                ty: BindingType::Buffer {
                    ty: BufferBindingType::Storage { read_only: false },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
            BindGroupLayoutEntry {
                binding: 1,
                visibility: ShaderStages::COMPUTE,
                ty: BindingType::StorageTexture {
                    access: StorageTextureAccess::ReadWrite,
                    format: TextureFormat::Rgba16Float,
                    view_dimension: TextureViewDimension::D2,
                },
                count: None,
            },
            BindGroupLayoutEntry {
                binding: 2,
                visibility: ShaderStages::COMPUTE,
                ty: BindingType::Buffer {
                    ty: BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
        ];

        let layout_descriptor = BindGroupLayoutDescriptor {
            label: "physarum_layout".into(),
            entries: entries.into(),
        };
        let layout = render_device
            .create_bind_group_layout(Some(&*layout_descriptor.label), &layout_descriptor.entries);

        pipeline.simulate_pipeline = Some(pipeline_cache.queue_compute_pipeline(
            ComputePipelineDescriptor {
                label: Some("physarum_simulate_pipeline".into()),
                layout: vec![layout_descriptor.clone()],
                push_constant_ranges: vec![],
                shader: resources.shader.clone(),
                shader_defs: vec![],
                entry_point: Some("simulate".into()),
                zero_initialize_workgroup_memory: false,
            },
        ));

        pipeline.diffuse_pipeline = Some(pipeline_cache.queue_compute_pipeline(
            ComputePipelineDescriptor {
                label: Some("physarum_diffuse_pipeline".into()),
                layout: vec![layout_descriptor],
                push_constant_ranges: vec![],
                shader: resources.shader.clone(),
                shader_defs: vec![],
                entry_point: Some("diffuse".into()),
                zero_initialize_workgroup_memory: false,
            },
        ));

        pipeline.bind_group_layout = Some(layout);
    }

    let Some(trail_map) = render_assets.get(&resources.trail_map) else {
        return;
    };
    let Some(agents_buffer) = render_buffers.get(&resources.agents) else {
        return;
    };
    let Some(simulate_id) = pipeline.simulate_pipeline else {
        return;
    };
    let Some(_) = pipeline_cache.get_compute_pipeline(simulate_id) else {
        return;
    };

    // Create a temporary buffer for the config uniform
    let mut config_buffer = UniformBuffer::from(config_res.config);
    config_buffer.write_buffer(&render_device, &render_queue);

    let bind_group = render_device.create_bind_group(
        "physarum_bind_group",
        pipeline.bind_group_layout.as_ref().unwrap(),
        &[
            BindGroupEntry {
                binding: 0,
                resource: agents_buffer.buffer.as_entire_binding(),
            },
            BindGroupEntry {
                binding: 1,
                resource: BindingResource::TextureView(&trail_map.texture_view),
            },
            BindGroupEntry {
                binding: 2,
                resource: config_buffer.buffer().unwrap().as_entire_binding(),
            },
        ],
    );

    commands.insert_resource(PhysarumBindGroup(bind_group));
}

/// Render the graph node that executes the Physarum compute shaders.
struct PhysarumNode;

impl render_graph::Node for PhysarumNode {
    fn run(
        &self,
        _graph: &mut render_graph::RenderGraphContext,
        render_context: &mut RenderContext,
        world: &World,
    ) -> Result<(), render_graph::NodeRunError> {
        let pipeline = world.resource::<PhysarumPipeline>();
        let Some(bind_group) = world.get_resource::<PhysarumBindGroup>() else {
            return Ok(());
        };
        let pipeline_cache = world.resource::<PipelineCache>();

        let Some(simulate_id) = pipeline.simulate_pipeline else {
            return Ok(());
        };
        let Some(diffuse_id) = pipeline.diffuse_pipeline else {
            return Ok(());
        };

        let Some(simulate_pipeline) = pipeline_cache.get_compute_pipeline(simulate_id) else {
            return Ok(());
        };
        let Some(diffuse_pipeline) = pipeline_cache.get_compute_pipeline(diffuse_id) else {
            return Ok(());
        };

        let mut pass =
            render_context
                .command_encoder()
                .begin_compute_pass(&ComputePassDescriptor {
                    label: Some("Physarum Compute Pass"),
                    ..default()
                });

        pass.set_bind_group(0, &bind_group.0, &[]);

        // Simulate
        pass.set_pipeline(simulate_pipeline);
        pass.dispatch_workgroups((AGENT_COUNT + 63) / 64, 1, 1);

        // Diffuse
        pass.set_pipeline(diffuse_pipeline);
        pass.dispatch_workgroups((SIZE.0 + 15) / 16, (SIZE.1 + 15) / 16, 1);

        Ok(())
    }
}
