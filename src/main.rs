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

#[derive(Message)]
struct RespawnAgentsEvent;

/// The max number of agents in the simulation.
const MAX_AGENT_COUNT: u32 = 2_000_000;
/// The resolution of the simulation and window.
const SIZE: (u32, u32) = (1920, 1080);

const DEFAULT_SENSOR_ANGLE: f32 = 35.0f32.to_radians();
const DEFAULT_SENSOR_DIST: f32 = 13.0;
const DEFAULT_TURN_SPEED: f32 = 550.0f32.to_radians();
const DEFAULT_MOVE_SPEED: f32 = 190.0;
const DEFAULT_AGENT_COUNT: u32 = 1_500_000;
const DEFAULT_DECAY: f32 = 1.0;
const DEFAULT_DIFFUSE_SPEED: f32 = 60.0;
const DEFAULT_DEPOSIT_AMOUNT: f32 = 0.007;
const DEFAULT_SPAWN_RADIUS: f32 = 0.55;
const DEFAULT_JITTER_AMOUNT: f32 = 0.1;

const DEFAULT_SPECIES_WEIGHTS: Vec4 = Vec4::new(1.0, 1.0, 1.0, 0.0);
const DEFAULT_INTERACTION_MATRIX: [Vec4; 4] = [
    Vec4::new(1.0, -1.0, -1.0, 0.0), // Species 0 (Red)
    Vec4::new(-1.0, 1.0, -1.0, 0.0), // Species 1 (Green)
    Vec4::new(-1.0, -1.0, 1.0, 0.0), // Species 2 (Blue)
    Vec4::ZERO,
];

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
                active_agents: DEFAULT_AGENT_COUNT,
                deposit_amount: DEFAULT_DEPOSIT_AMOUNT,
                spawn_radius: DEFAULT_SPAWN_RADIUS,
                jitter_amount: DEFAULT_JITTER_AMOUNT,
                _padding1: 0.0,
                _padding2: 0.0,
                _padding3: 0.0,
                interaction_matrix: DEFAULT_INTERACTION_MATRIX,
                species_weights: DEFAULT_SPECIES_WEIGHTS,
            },
        })
        .add_message::<RespawnAgentsEvent>()
        .add_systems(Startup, setup)
        .add_systems(Update, (update_config, move_fps_overlay, handle_respawn))
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
    /// The species ID of the agent (0, 1, or 2).
    species: u32,
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
    /// The number of agents to simulate.
    active_agents: u32,
    /// Amount of pheromone an agent deposits each step.
    deposit_amount: f32,
    /// Radius of the spawn clusters as a percentage of the smallest window dimension.
    spawn_radius: f32,
    /// Amount of random jitter added during trail tracking.
    jitter_amount: f32,
    /// Padding to align species_weights to 16 bytes and satisfy Pod/Zeroable.
    _padding1: f32,
    _padding2: f32,
    _padding3: f32,
    /// Weights for species distribution (Red, Green, Blue, _unused).
    species_weights: Vec4,
    /// Matrix defining how each species (rows) interacts with each color channel (columns: R, G, B, A).
    interaction_matrix: [Vec4; 4],
}

/// Generates a vector of agents with positions and species distributed according to the configuration.
fn generate_agents(config: &PhysarumConfig) -> Vec<Agent> {
    let mut rng = rand::rng();
    let total_weight = config.species_weights.x + config.species_weights.y + config.species_weights.z;
    let w0 = if total_weight > 0.0 {
        config.species_weights.x / total_weight
    } else {
        1.0 / 3.0
    };
    let w1 = if total_weight > 0.0 {
        config.species_weights.y / total_weight
    } else {
        1.0 / 3.0
    };

    let width = config.width as f32;
    let height = config.height as f32;
    let centers = [
        Vec2::new(width / 2.0, height / 4.0),       // Species 0
        Vec2::new(width / 4.0, height * 3.0 / 4.0), // Species 1
        Vec2::new(width * 3.0 / 4.0, height * 3.0 / 4.0), // Species 2
    ];
    let spawn_radius = config.spawn_radius * width.min(height);

    (0..MAX_AGENT_COUNT)
        .map(|_| {
            let angle = rng.random_range(0.0..std::f32::consts::TAU);
            let r_species = rng.random_range(0.0..1.0);
            let species = if r_species < w0 {
                0
            } else if r_species < w0 + w1 {
                1
            } else {
                2
            };

            // Generate uniform random position within a circle cluster
            let r = spawn_radius * rng.random_range(0.0..1.0f32).sqrt();
            let theta = rng.random_range(0.0..std::f32::consts::TAU);
            let pos = centers[species] + Vec2::new(theta.cos() * r, theta.sin() * r);

            Agent {
                pos,
                angle,
                species: species as u32,
            }
        })
        .collect()
}

/// Initializes the simulation resources, agents, and camera.
fn setup(
    mut commands: Commands,
    mut images: ResMut<Assets<Image>>,
    mut buffers: ResMut<Assets<ShaderStorageBuffer>>,
    asset_server: Res<AssetServer>,
    config_res: Res<PhysarumConfigResource>,
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
        bevy::asset::RenderAssetUsages::default(),
    );
    image.texture_descriptor.usage |=
        TextureUsages::STORAGE_BINDING | TextureUsages::TEXTURE_BINDING;
    let trail_map_handle = images.add(image);

    // Initialize agents
    let agents_data = generate_agents(&config_res.config);
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

/// System that handles respawning agents with a new species distribution.
fn handle_respawn(
    mut events: MessageReader<RespawnAgentsEvent>,
    config_res: Res<PhysarumConfigResource>,
    resources: Res<PhysarumResources>,
    mut buffers: ResMut<Assets<ShaderStorageBuffer>>,
    mut images: ResMut<Assets<Image>>,
) {
    if events.is_empty() {
        return;
    }
    events.clear();

    let agents_data = generate_agents(&config_res.config);

    if let Some(buffer) = buffers.get_mut(&resources.agents) {
        *buffer = ShaderStorageBuffer::from(agents_data);
    }

    if let Some(image) = images.get_mut(&resources.trail_map) {
        if let Some(data) = &mut image.data {
            data.chunks_exact_mut(8).for_each(|chunk| {
                chunk.copy_from_slice(&[0, 0, 0, 0, 0, 0, 0, 60]);
            });
        }
    }
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
fn physarum_ui(
    mut contexts: EguiContexts,
    mut config_res: ResMut<PhysarumConfigResource>,
    mut respawn_events: MessageWriter<RespawnAgentsEvent>,
) {
    let ctx = match contexts.ctx_mut() {
        Ok(ctx) => ctx,
        Err(_) => return,
    };

    egui::Window::new("Physarum Config")
        .default_open(true)
        .show(ctx, |ui| {
            let config = &mut config_res.config;

            egui::Grid::new("config_grid")
                .num_columns(3)
                .spacing([10.0, 8.0])
                .show(ui, |ui| {
                    // Sensor Angle
                    ui.label("Sensor Angle")
                        .on_hover_text("The angle (in degrees) at which the left and right sensors are offset from the agent's forward direction.");
                    let mut sensor_angle_deg = config.sensor_angle.to_degrees();
                    let slider_res = ui.add_sized(
                        [140.0, 20.0],
                        egui::Slider::new(&mut sensor_angle_deg, 0.0..=180.0).show_value(false),
                    );
                    let drag_res = ui.add_sized(
                        [60.0, 20.0],
                        egui::DragValue::new(&mut sensor_angle_deg)
                            .speed(0.1)
                            .suffix("°"),
                    );
                    if slider_res.changed() || drag_res.changed() {
                        config.sensor_angle = sensor_angle_deg.to_radians();
                    }
                    ui.end_row();

                    // Sensor Distance
                    ui.label("Sensor Dist")
                        .on_hover_text("The distance (in pixels) from the agent to its sensors.");
                    ui.add_sized(
                        [140.0, 20.0],
                        egui::Slider::new(&mut config.sensor_dist, 0.0..=100.0).show_value(false),
                    );
                    ui.add_sized(
                        [60.0, 20.0],
                        egui::DragValue::new(&mut config.sensor_dist).speed(0.1),
                    );
                    ui.end_row();

                    // Turn Speed
                    ui.label("Turn Speed")
                        .on_hover_text("The speed at which the agent turns towards pheromones (in degrees per second).");
                    let mut turn_speed_deg = config.turn_speed.to_degrees();
                    let slider_res = ui.add_sized(
                        [140.0, 20.0],
                        egui::Slider::new(&mut turn_speed_deg, 0.0..=3600.0).show_value(false),
                    );
                    let drag_res = ui.add_sized(
                        [60.0, 20.0],
                        egui::DragValue::new(&mut turn_speed_deg)
                            .speed(1.0)
                            .suffix("°"),
                    );
                    if slider_res.changed() || drag_res.changed() {
                        config.turn_speed = turn_speed_deg.to_radians();
                    }
                    ui.end_row();

                    // Move Speed
                    ui.label("Max Move Speed")
                        .on_hover_text("The maximum speed of the agents. They move slower when sensing less pheromone (min 20% of max).");
                    ui.add_sized(
                        [140.0, 20.0],
                        egui::Slider::new(&mut config.move_speed, 0.0..=500.0).show_value(false),
                    );
                    ui.add_sized(
                        [60.0, 20.0],
                        egui::DragValue::new(&mut config.move_speed).speed(1.0),
                    );
                    ui.end_row();

                    // Active Agents
                    ui.label("Agents")
                        .on_hover_text("The number of agents to simulate (up to the maximum capacity).");
                    ui.add_sized(
                        [140.0, 20.0],
                        egui::Slider::new(&mut config.active_agents, 0..=MAX_AGENT_COUNT).show_value(false),
                    );
                    ui.add_sized(
                        [60.0, 20.0],
                        egui::DragValue::new(&mut config.active_agents).speed(1000.0),
                    );
                    ui.end_row();

                    // Evaporation
                    ui.label("Evap Time")
                        .on_hover_text("The number of seconds it takes for a trail to fully evaporate (linear decay).");
                    let mut evap_time = if config.decay > 0.0 { 1.0 / config.decay } else { 10.0 };
                    let slider_res = ui.add_sized(
                        [140.0, 20.0],
                        egui::Slider::new(&mut evap_time, 0.1..=10.0).show_value(false),
                    );
                    let drag_res = ui.add_sized(
                        [60.0, 20.0],
                        egui::DragValue::new(&mut evap_time)
                            .speed(0.1)
                            .suffix("s"),
                    );
                    if slider_res.changed() || drag_res.changed() {
                        config.decay = 1.0 / evap_time;
                    }
                    ui.end_row();

                    // Deposit Amount
                    ui.label("Deposit Amount")
                        .on_hover_text("The amount of pheromone an agent deposits each step.");
                    ui.add_sized(
                        [140.0, 20.0],
                        egui::Slider::new(&mut config.deposit_amount, 0.0..=0.05).show_value(false),
                    );
                    ui.add_sized(
                        [60.0, 20.0],
                        egui::DragValue::new(&mut config.deposit_amount).speed(0.01),
                    );
                    ui.end_row();

                    // Spawn Radius
                    ui.label("Spawn Radius")
                        .on_hover_text("The radius of the spawn clusters as a percentage of the smallest window dimension.");
                    let mut spawn_radius_pct = config.spawn_radius * 100.0;
                    let slider_res = ui.add_sized(
                        [140.0, 20.0],
                        egui::Slider::new(&mut spawn_radius_pct, 0.0..=100.0).show_value(false),
                    );
                    ui.add_sized(
                        [60.0, 20.0],
                        egui::DragValue::new(&mut spawn_radius_pct)
                            .speed(0.1)
                            .suffix("%"),
                    );
                    if slider_res.changed() || drag_res.changed() {
                        config.spawn_radius = spawn_radius_pct / 100.0;
                    }
                    ui.end_row();

                    // Jitter Amount
                    ui.label("Jitter Amount")
                        .on_hover_text("The amount of random jitter added during trail tracking to encourage branching.");
                    ui.add_sized(
                        [140.0, 20.0],
                        egui::Slider::new(&mut config.jitter_amount, 0.0..=1.0).show_value(false),
                    );
                    ui.add_sized(
                        [60.0, 20.0],
                        egui::DragValue::new(&mut config.jitter_amount).speed(0.01),
                    );
                    ui.end_row();

                    // Diffusion Speed
                    ui.label("Diffusion Speed")
                        .on_hover_text("The speed at which pheromones diffuse into neighboring areas.");
                    ui.add_sized(
                        [140.0, 20.0],
                        egui::Slider::new(&mut config.diffuse_speed, 0.0..=200.0).show_value(false),
                    );
                    ui.add_sized(
                        [60.0, 20.0],
                        egui::DragValue::new(&mut config.diffuse_speed).speed(1.0),
                    );
                    ui.end_row();
                });

            ui.add_space(20.0);
            ui.separator();
            ui.heading("Species Distribution");
            ui.add_space(5.0);
            ui.label("Relative weights for spawning each species.");

            egui::Grid::new("distribution_grid")
                .num_columns(2)
                .spacing([10.0, 8.0])
                .show(ui, |ui| {
                    ui.label("Red");
                    ui.add(egui::Slider::new(&mut config.species_weights.x, 0.0..=1.0));
                    ui.end_row();
                    ui.label("Green");
                    ui.add(egui::Slider::new(&mut config.species_weights.y, 0.0..=1.0));
                    ui.end_row();
                    ui.label("Blue");
                    ui.add(egui::Slider::new(&mut config.species_weights.z, 0.0..=1.0));
                    ui.end_row();
                });

            ui.add_space(5.0);
            if ui.button("Respawn Agents").clicked() {
                respawn_events.write(RespawnAgentsEvent);
            }

            ui.add_space(20.0);
            ui.separator();
            ui.heading("Species Interaction");
            ui.add_space(5.0);
            ui.label("Attraction/Repulsion weights for each species (row) towards different pheromone (column).");
            ui.add_space(5.0);

            let species_names = ["Red", "Green", "Blue"];
            egui::Grid::new("interaction_grid")
                .num_columns(4)
                .spacing([10.0, 8.0])
                .show(ui, |ui| {
                    ui.label("");
                    ui.label("Red").on_hover_text("Red pheromone sensed.");
                    ui.label("Green").on_hover_text("Green pheromone sensed.");
                    ui.label("Blue").on_hover_text("Blue pheromone sensed.");
                    ui.end_row();

                    for i in 0..3 {
                        ui.label(species_names[i]);
                        for j in 0..3 {
                            let val = match j {
                                0 => &mut config.interaction_matrix[i].x,
                                1 => &mut config.interaction_matrix[i].y,
                                2 => &mut config.interaction_matrix[i].z,
                                _ => unreachable!(),
                            };
                            ui.add(egui::DragValue::new(val).speed(0.01).range(-1.0..=1.0));
                        }
                        ui.end_row();
                    }
                });

            ui.add_space(20.0);
            if ui.button("Reset to Defaults").clicked() {
                config.sensor_angle = DEFAULT_SENSOR_ANGLE;
                config.sensor_dist = DEFAULT_SENSOR_DIST;
                config.turn_speed = DEFAULT_TURN_SPEED;
                config.move_speed = DEFAULT_MOVE_SPEED;
                config.decay = DEFAULT_DECAY;
                config.active_agents = DEFAULT_AGENT_COUNT;
                config.deposit_amount = DEFAULT_DEPOSIT_AMOUNT;
                config.spawn_radius = DEFAULT_SPAWN_RADIUS;
                config.jitter_amount = DEFAULT_JITTER_AMOUNT;
                config.species_weights = DEFAULT_SPECIES_WEIGHTS;
                config.interaction_matrix = DEFAULT_INTERACTION_MATRIX;
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

        let active_agents = world.resource::<PhysarumConfigResource>().config.active_agents;

        // Simulate
        pass.set_pipeline(simulate_pipeline);
        pass.dispatch_workgroups((active_agents + 63) / 64, 1, 1);

        // Diffuse
        pass.set_pipeline(diffuse_pipeline);
        pass.dispatch_workgroups((SIZE.0 + 15) / 16, (SIZE.1 + 15) / 16, 1);

        Ok(())
    }
}
