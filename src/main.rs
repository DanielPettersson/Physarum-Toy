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
use bytemuck::{Pod, Zeroable};
use rand::RngExt;

const AGENT_COUNT: u32 = 200_000;
const SIZE: (u32, u32) = (1280, 720);

fn main() {
    App::new()
        .add_plugins(DefaultPlugins.set(WindowPlugin {
            primary_window: Some(Window {
                title: "Physarum Toy".to_string(),
                ..default()
            }),
            ..default()
        }))
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
        .insert_resource(PhysarumConfigResource {
            config: PhysarumConfig {
                sensor_angle: 0.35,
                sensor_dist: 15.0,
                turn_speed: 10.0,
                move_speed: 50.0,
                decay: 0.9,
                width: SIZE.0,
                height: SIZE.1,
                delta_time: 0.0,
            },
        })
        .add_systems(Startup, setup)
        .add_systems(Update, (update_config, move_fps_overlay))
        .add_plugins(PhysarumComputePlugin)
        .run();
}

#[derive(Resource, Clone, ExtractResource)]
struct PhysarumResources {
    agents: Handle<ShaderStorageBuffer>,
    trail_map: Handle<Image>,
    shader: Handle<Shader>,
}

#[derive(Resource, ExtractResource, Clone)]
struct PhysarumConfigResource {
    config: PhysarumConfig,
}

#[repr(C)]
#[derive(Copy, Clone, Debug, Pod, Zeroable, ShaderType)]
struct Agent {
    pos: Vec2,
    angle: f32,
    _pad: f32,
}

#[repr(C)]
#[derive(Copy, Clone, Debug, Pod, Zeroable, ShaderType)]
struct PhysarumConfig {
    sensor_angle: f32,
    sensor_dist: f32,
    turn_speed: f32,
    move_speed: f32,
    decay: f32,
    width: u32,
    height: u32,
    delta_time: f32,
}

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
        &[0, 0, 0, 255],
        TextureFormat::Rgba8Unorm,
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

fn update_config(time: Res<Time>, mut config_res: ResMut<PhysarumConfigResource>) {
    config_res.config.delta_time = time.delta_secs();
}

fn move_fps_overlay(mut query: Query<(&mut Node, &GlobalZIndex)>) {
    for (mut node, z_index) in &mut query {
        if z_index.0 == i32::MAX - 32 {
            node.top = Val::Auto;
            node.bottom = Val::Px(10.0);
            node.left = Val::Px(10.0);
        }
    }
}

// --- Compute Infrastructure ---

struct PhysarumComputePlugin;

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

#[derive(Resource, Default)]
struct PhysarumPipeline {
    simulate_pipeline: Option<CachedComputePipelineId>,
    diffuse_pipeline: Option<CachedComputePipelineId>,
    bind_group_layout: Option<BindGroupLayout>,
}

#[derive(Resource)]
struct PhysarumBindGroup(BindGroup);

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
                    format: TextureFormat::Rgba8Unorm,
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
        pass.dispatch_workgroups(SIZE.0 / 16, SIZE.1 / 16, 1);

        Ok(())
    }
}
