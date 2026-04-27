//! # Physarum Toy
//!
//! A GPU-accelerated simulation of Physarum polycephalum (slime mold) using Bevy and WGSL compute shaders.

mod simulation;
mod ui;
mod compute;

use bevy::{
    dev_tools::fps_overlay::{FpsOverlayConfig, FpsOverlayPlugin, FrameTimeGraphConfig},
    prelude::*,
    render::extract_resource::ExtractResourcePlugin,
    window::WindowResolution,
};
use bevy_egui::{EguiPlugin, EguiPrimaryContextPass};
use simulation::*;
use ui::*;
use compute::*;

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
        .init_resource::<UiState>()
        .add_systems(Startup, setup)
        .add_systems(Update, (update_config, handle_window_resize, move_fps_overlay, handle_respawn, toggle_ui))
        .add_systems(EguiPrimaryContextPass, physarum_ui)
        .add_plugins(PhysarumComputePlugin)
        .run();
}
