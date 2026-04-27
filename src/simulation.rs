use bevy::{
    prelude::*,
    render::{
        extract_resource::ExtractResource,
        render_resource::*,
        storage::ShaderStorageBuffer,
    },
    window::WindowResized,
};
use bytemuck::{Pod, Zeroable};
use rand::RngExt;

#[derive(Message)]
pub struct RespawnAgentsEvent;

/// The max number of agents in the simulation.
pub const MAX_AGENT_COUNT: u32 = 4_000_000;
/// The resolution of the simulation and window.
pub const SIZE: (u32, u32) = (1920, 1080);

pub const DEFAULT_SENSOR_ANGLE: f32 = 35.0f32.to_radians();
pub const DEFAULT_SENSOR_DIST: f32 = 13.0;
pub const DEFAULT_TURN_SPEED: f32 = 550.0f32.to_radians();
pub const DEFAULT_MOVE_SPEED: f32 = 190.0;
pub const DEFAULT_AGENT_COUNT: u32 = 1_500_000;
pub const DEFAULT_DECAY: f32 = 2.0;
pub const DEFAULT_DIFFUSE_SPEED: f32 = 30.0;
pub const DEFAULT_DEPOSIT_AMOUNT: f32 = 0.007;
pub const DEFAULT_SPAWN_RADIUS: f32 = 0.55;
pub const DEFAULT_JITTER_AMOUNT: f32 = 0.1;

pub const DEFAULT_SPECIES_WEIGHTS: Vec4 = Vec4::new(1.0, 1.0, 1.0, 0.0);
pub const DEFAULT_INTERACTION_MATRIX: [Vec4; 4] = [
    Vec4::new(1.0, -1.0, -1.0, 0.0), // Species 0 (Red)
    Vec4::new(-1.0, 1.0, -1.0, 0.0), // Species 1 (Green)
    Vec4::new(-1.0, -1.0, 1.0, 0.0), // Species 2 (Blue)
    Vec4::ZERO,
];

/// Resources required for the Physarum simulation on the GPU.
#[derive(Resource, Clone, ExtractResource)]
pub struct PhysarumResources {
    /// Buffer containing all agent agents.
    pub agents: Handle<ShaderStorageBuffer>,
    /// The trail map texture where pheromones are deposited and sensed.
    pub trail_map: Handle<Image>,
    /// Temporary trail map for the two-pass diffusion.
    pub trail_map_temp: Handle<Image>,
    /// The compute shader handle.
    pub shader: Handle<Shader>,
}

/// A wrapper resource for the simulation configuration, allowing it to be extracted to the render world.
#[derive(Resource, ExtractResource, Clone)]
pub struct PhysarumConfigResource {
    pub config: PhysarumConfig,
}

/// Representation of a single agent.
#[repr(C)]
#[derive(Copy, Clone, Debug, Pod, Zeroable, ShaderType)]
pub struct Agent {
    /// 2D position in the simulation space.
    pub pos: Vec2,
    /// Orientation angle in radians.
    pub angle: f32,
    /// The species ID of the agent (0, 1, or 2).
    pub species: u32,
}

/// Configuration parameters for the Physarum simulation.
#[repr(C)]
#[derive(Copy, Clone, Debug, Pod, Zeroable, ShaderType)]
pub struct PhysarumConfig {
    /// Angle at which the sensors are offset from the agent's forward direction.
    pub sensor_angle: f32,
    /// Distance from the agent to its sensors.
    pub sensor_dist: f32,
    /// Speed at which the agent turns towards pheromones.
    pub turn_speed: f32,
    /// Speed at which the agent moves forward.
    pub move_speed: f32,
    /// Rate of pheromone decay over time.
    pub decay: f32,
    /// Width of the simulation area.
    pub width: u32,
    /// Height of the simulation area.
    pub height: u32,
    /// Time elapsed since the last frame.
    pub delta_time: f32,
    /// Speed of pheromone diffusion.
    pub diffuse_speed: f32,
    /// The number of agents to simulate.
    pub active_agents: u32,
    /// Amount of pheromone an agent deposits each step.
    pub deposit_amount: f32,
    /// Radius of the spawn clusters as a percentage of the smallest window dimension.
    pub spawn_radius: f32,
    /// Amount of random jitter added during trail tracking.
    pub jitter_amount: f32,
    /// Padding to align species_weights to 16 bytes and satisfy Pod/Zeroable.
    pub _padding1: f32,
    pub _padding2: f32,
    pub _padding3: f32,
    /// Weights for species distribution (Red, Green, Blue, _unused).
    pub species_weights: Vec4,
    /// Matrix defining how each species (rows) interacts with each color channel (columns: R, G, B, A).
    pub interaction_matrix: [Vec4; 4],
}

/// Generates a vector of agents with positions and species distributed according to the configuration.
pub fn generate_agents(config: &PhysarumConfig) -> Vec<Agent> {
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
pub fn setup(
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
    
    let trail_map_handle = images.add(image.clone());
    let trail_map_temp_handle = images.add(image);

    // Initialize agents
    let agents_data = generate_agents(&config_res.config);
    let agents_buffer = buffers.add(ShaderStorageBuffer::from(agents_data));
    let shader = asset_server.load("shaders/physarum.wgsl");

    commands.insert_resource(PhysarumResources {
        agents: agents_buffer,
        trail_map: trail_map_handle.clone(),
        trail_map_temp: trail_map_temp_handle,
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
pub fn update_config(time: Res<Time>, mut config_res: ResMut<PhysarumConfigResource>) {
    config_res.config.delta_time = time.delta_secs();
}

/// Updates the simulation configuration and resources when the window is resized.
pub fn handle_window_resize(
    mut resize_reader: MessageReader<WindowResized>,
    mut config_res: ResMut<PhysarumConfigResource>,
    mut images: ResMut<Assets<Image>>,
    resources: Res<PhysarumResources>,
    mut query: Query<&mut Sprite>,
) {
    for e in resize_reader.read() {
        let new_width = e.width as u32;
        let new_height = e.height as u32;

        if new_width == 0 || new_height == 0 {
            continue;
        }

        config_res.config.width = new_width;
        config_res.config.height = new_height;

        // Resize the trail map images
        if let Some(image) = images.get_mut(&resources.trail_map) {
            image.resize(Extent3d {
                width: new_width,
                height: new_height,
                depth_or_array_layers: 1,
            });
        }
        if let Some(image) = images.get_mut(&resources.trail_map_temp) {
            image.resize(Extent3d {
                width: new_width,
                height: new_height,
                depth_or_array_layers: 1,
            });
        }

        // Update the sprite size to match the new window size
        for mut sprite in query.iter_mut() {
            sprite.custom_size = Some(Vec2::new(new_width as f32, new_height as f32));
        }
    }
}

/// System that handles respawning agents with a new species distribution.
pub fn handle_respawn(
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
    if let Some(image) = images.get_mut(&resources.trail_map_temp) {
        if let Some(data) = &mut image.data {
            data.chunks_exact_mut(8).for_each(|chunk| {
                chunk.copy_from_slice(&[0, 0, 0, 0, 0, 0, 0, 60]);
            });
        }
    }
}

/// Adjusts the position of the FPS overlay to the bottom-left corner.
pub fn move_fps_overlay(mut query: Query<(&mut Node, &GlobalZIndex)>) {
    for (mut node, z_index) in &mut query {
        if z_index.0 == i32::MAX - 32 {
            node.top = Val::Auto;
            node.bottom = Val::Px(10.0);
            node.left = Val::Px(10.0);
        }
    }
}
