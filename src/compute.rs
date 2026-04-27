use bevy::{
    prelude::*,
    render::{
        Render, RenderApp,
        render_asset::RenderAssets,
        render_graph::{self, RenderGraph, RenderLabel},
        render_resource::*,
        renderer::{RenderContext, RenderDevice, RenderQueue},
        storage::GpuShaderStorageBuffer,
        texture::GpuImage,
    },
};
use crate::simulation::*;

// --- Compute Infrastructure ---

/// Plugin responsible for setting up the compute shader pipeline and render graph nodes.
pub struct PhysarumComputePlugin;

/// Label identifying the Physarum compute node in the render graph.
#[derive(RenderLabel, Debug, Hash, PartialEq, Eq, Clone)]
pub struct PhysarumLabel;

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
pub struct PhysarumPipeline {
    /// Pipeline for the agent simulation step.
    pub simulate_pipeline: Option<CachedComputePipelineId>,
    /// Pipeline for the horizontal diffusion pass.
    pub diffuse_h_pipeline: Option<CachedComputePipelineId>,
    /// Pipeline for the vertical diffusion pass.
    pub diffuse_v_pipeline: Option<CachedComputePipelineId>,
    /// The common bind group layout used by both pipelines.
    pub bind_group_layout: Option<BindGroupLayout>,
    /// Persistent uniform buffer for configuration.
    pub config_buffer: Option<UniformBuffer<PhysarumConfig>>,
}

/// Wrapper for the GPU bind group used in the compute shader.
#[derive(Resource)]
pub struct PhysarumBindGroup(pub BindGroup);

/// Prepares the bind group and initializes pipelines for the compute shader.
pub fn prepare_bind_group(
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
            BindGroupLayoutEntry {
                binding: 3,
                visibility: ShaderStages::COMPUTE,
                ty: BindingType::StorageTexture {
                    access: StorageTextureAccess::ReadWrite,
                    format: TextureFormat::Rgba16Float,
                    view_dimension: TextureViewDimension::D2,
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

        pipeline.diffuse_h_pipeline = Some(pipeline_cache.queue_compute_pipeline(
            ComputePipelineDescriptor {
                label: Some("physarum_diffuse_h_pipeline".into()),
                layout: vec![layout_descriptor.clone()],
                push_constant_ranges: vec![],
                shader: resources.shader.clone(),
                shader_defs: vec![],
                entry_point: Some("diffuse_h".into()),
                zero_initialize_workgroup_memory: false,
            },
        ));

        pipeline.diffuse_v_pipeline = Some(pipeline_cache.queue_compute_pipeline(
            ComputePipelineDescriptor {
                label: Some("physarum_diffuse_v_pipeline".into()),
                layout: vec![layout_descriptor],
                push_constant_ranges: vec![],
                shader: resources.shader.clone(),
                shader_defs: vec![],
                entry_point: Some("diffuse_v".into()),
                zero_initialize_workgroup_memory: false,
            },
        ));

        pipeline.bind_group_layout = Some(layout);
        pipeline.config_buffer = Some(UniformBuffer::from(config_res.config));
    }

    let Some(trail_map) = render_assets.get(&resources.trail_map) else {
        return;
    };
    let Some(trail_map_temp) = render_assets.get(&resources.trail_map_temp) else {
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

    let PhysarumPipeline {
        bind_group_layout,
        config_buffer,
        ..
    } = &mut *pipeline;

    let layout = bind_group_layout.as_ref().unwrap();
    let config_buffer = config_buffer.as_mut().unwrap();

    // Update the existing uniform buffer
    config_buffer.set(config_res.config);
    config_buffer.write_buffer(&render_device, &render_queue);
    let config_buffer_binding = config_buffer.buffer().unwrap().as_entire_binding();

    let bind_group = render_device.create_bind_group(
        "physarum_bind_group",
        layout,
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
                resource: config_buffer_binding,
            },
            BindGroupEntry {
                binding: 3,
                resource: BindingResource::TextureView(&trail_map_temp.texture_view),
            },
        ],
    );

    commands.insert_resource(PhysarumBindGroup(bind_group));
}

/// Render the graph node that executes the Physarum compute shaders.
pub struct PhysarumNode;

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
        let Some(diffuse_h_id) = pipeline.diffuse_h_pipeline else {
            return Ok(());
        };
        let Some(diffuse_v_id) = pipeline.diffuse_v_pipeline else {
            return Ok(());
        };

        let Some(simulate_pipeline) = pipeline_cache.get_compute_pipeline(simulate_id) else {
            return Ok(());
        };
        let Some(diffuse_h_pipeline) = pipeline_cache.get_compute_pipeline(diffuse_h_id) else {
            return Ok(());
        };
        let Some(diffuse_v_pipeline) = pipeline_cache.get_compute_pipeline(diffuse_v_id) else {
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
        let width = world.resource::<PhysarumConfigResource>().config.width;
        let height = world.resource::<PhysarumConfigResource>().config.height;

        // Simulate
        pass.set_pipeline(simulate_pipeline);
        pass.dispatch_workgroups((active_agents + 63) / 64, 1, 1);

        // Diffuse Horizontal
        pass.set_pipeline(diffuse_h_pipeline);
        pass.dispatch_workgroups((width + 15) / 16, (height + 15) / 16, 1);

        // Diffuse Vertical
        pass.set_pipeline(diffuse_v_pipeline);
        pass.dispatch_workgroups((width + 15) / 16, (height + 15) / 16, 1);

        Ok(())
    }
}
