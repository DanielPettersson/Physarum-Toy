use bevy::prelude::*;
use bevy_egui::{egui, EguiContexts};
use crate::simulation::*;
use rand::RngExt;

#[derive(Resource)]
pub struct UiState {
    pub visible: bool,
}

impl Default for UiState {
    fn default() -> Self {
        Self { visible: true }
    }
}

pub fn toggle_ui(
    keyboard_input: Res<ButtonInput<KeyCode>>,
    mut ui_state: ResMut<UiState>,
) {
    if keyboard_input.just_pressed(KeyCode::Space) || keyboard_input.just_pressed(KeyCode::KeyH) {
        ui_state.visible = !ui_state.visible;
    }
}

/// System that draws the configuration UI.
pub fn physarum_ui(
    mut contexts: EguiContexts,
    mut config_res: ResMut<PhysarumConfigResource>,
    mut respawn_events: MessageWriter<RespawnAgentsEvent>,
    ui_state: Res<UiState>,
) {
    let ctx = match contexts.ctx_mut() {
        Ok(ctx) => ctx,
        Err(_) => return,
    };

    if !ui_state.visible {
        egui::Area::new("ui_hint".into())
            .anchor(egui::Align2::LEFT_TOP, [10.0, 10.0])
            .show(ctx, |ui| {
                ui.label(egui::RichText::new("Press Space or H to show UI").small().color(egui::Color32::GRAY));
            });
        return;
    }

    egui::Window::new("Physarum Config")
        .default_open(true)
        .default_width(320.0)
        .resizable(true)
        .show(ctx, |ui| {
            egui::ScrollArea::vertical()
                .max_height(600.0) // Prevent the window from becoming too tall
                .show(ui, |ui| {
                let config = &mut config_res.config;

                // --- Agent Behavior ---
                ui.collapsing("Agent Behavior", |ui| {
                    egui::Grid::new("agent_behavior_grid")
                        .num_columns(2)
                        .spacing([10.0, 8.0])
                        .show(ui, |ui| {
                            // Sensor Angle
                            ui.label("Sensor Angle").on_hover_text("Angle of left/right sensors from forward direction.");
                            let mut sensor_angle_deg = config.sensor_angle.to_degrees();
                            if ui.add(egui::Slider::new(&mut sensor_angle_deg, 0.0..=180.0).suffix("°")).changed() {
                                config.sensor_angle = sensor_angle_deg.to_radians();
                            }
                            ui.end_row();

                            // Sensor Distance
                            ui.label("Sensor Dist").on_hover_text("Distance from agent to its sensors.");
                            ui.add(egui::Slider::new(&mut config.sensor_dist, 0.0..=100.0));
                            ui.end_row();

                            // Turn Speed
                            ui.label("Turn Speed").on_hover_text("Speed of turning towards pheromones.");
                            let mut turn_speed_deg = config.turn_speed.to_degrees();
                            if ui.add(egui::Slider::new(&mut turn_speed_deg, 0.0..=3600.0).suffix("°/s")).changed() {
                                config.turn_speed = turn_speed_deg.to_radians();
                            }
                            ui.end_row();

                            // Move Speed
                            ui.label("Move Speed").on_hover_text("Maximum speed of the agents.");
                            ui.add(egui::Slider::new(&mut config.move_speed, 0.0..=500.0));
                            ui.end_row();

                            // Jitter Amount
                            ui.label("Jitter").on_hover_text("Randomness added to trail tracking.");
                            ui.add(egui::Slider::new(&mut config.jitter_amount, 0.0..=1.0));
                            ui.end_row();
                        });
                });

                ui.add_space(5.0);

                // --- Environment ---
                ui.collapsing("Environment", |ui| {
                    egui::Grid::new("environment_grid")
                        .num_columns(2)
                        .spacing([10.0, 8.0])
                        .show(ui, |ui| {
                            // Evaporation
                            ui.label("Evap Time").on_hover_text("Time for a trail to fully evaporate.");
                            let mut evap_time = if config.decay > 0.0 { 1.0 / config.decay } else { 10.0 };
                            if ui.add(egui::Slider::new(&mut evap_time, 0.1..=10.0).suffix("s")).changed() {
                                config.decay = 1.0 / evap_time;
                            }
                            ui.end_row();

                            // Deposit Amount
                            ui.label("Deposit").on_hover_text("Pheromone deposited per step.");
                            ui.add(egui::Slider::new(&mut config.deposit_amount, 0.0..=0.05));
                            ui.end_row();

                            // Diffusion Speed
                            ui.label("Diffusion").on_hover_text("Speed of pheromone diffusion.");
                            ui.add(egui::Slider::new(&mut config.diffuse_speed, 0.0..=200.0));
                            ui.end_row();
                        });
                });

                ui.add_space(5.0);

                // --- Spawning ---
                ui.collapsing("Spawning", |ui| {
                    egui::Grid::new("spawning_grid")
                        .num_columns(2)
                        .spacing([10.0, 8.0])
                        .show(ui, |ui| {
                            // Active Agents
                            ui.label("Active Agents").on_hover_text("Number of agents currently active.");
                            ui.add(egui::Slider::new(&mut config.active_agents, 0..=MAX_AGENT_COUNT).logarithmic(true));
                            ui.end_row();

                            // Spawn Radius
                            ui.label("Spawn Radius").on_hover_text("Radius of species clusters.");
                            let mut spawn_radius_pct = config.spawn_radius * 100.0;
                            if ui.add(egui::Slider::new(&mut spawn_radius_pct, 0.0..=100.0).suffix("%")).changed() {
                                config.spawn_radius = spawn_radius_pct / 100.0;
                            }
                            ui.end_row();
                        });

                    ui.add_space(5.0);
                    ui.label("Species Weight Distribution:");
                    egui::Grid::new("species_weights_grid")
                        .num_columns(2)
                        .spacing([10.0, 8.0])
                        .show(ui, |ui| {
                            ui.colored_label(egui::Color32::from_rgb(255, 100, 100), "Red");
                            ui.add(egui::Slider::new(&mut config.species_weights.x, 0.0..=1.0));
                            ui.end_row();

                            ui.colored_label(egui::Color32::from_rgb(100, 255, 100), "Green");
                            ui.add(egui::Slider::new(&mut config.species_weights.y, 0.0..=1.0));
                            ui.end_row();

                            ui.colored_label(egui::Color32::from_rgb(100, 100, 255), "Blue");
                            ui.add(egui::Slider::new(&mut config.species_weights.z, 0.0..=1.0));
                            ui.end_row();
                        });

                    ui.add_space(10.0);
                    if ui.button("Respawn Agents").clicked() {
                        respawn_events.write(RespawnAgentsEvent);
                    }
                });

                ui.add_space(5.0);

                // --- Species Interaction ---
                ui.collapsing("Species Interaction", |ui| {
                    ui.label("How species (rows) react to pheromones (cols):");
                    ui.add_space(5.0);

                    let species_names = ["Red", "Green", "Blue"];
                    let species_colors = [
                        egui::Color32::from_rgb(255, 100, 100),
                        egui::Color32::from_rgb(100, 255, 100),
                        egui::Color32::from_rgb(100, 100, 255),
                    ];

                    egui::Grid::new("interaction_grid")
                        .num_columns(4)
                        .spacing([10.0, 8.0])
                        .show(ui, |ui| {
                            ui.label("");
                            for i in 0..3 {
                                ui.colored_label(species_colors[i], species_names[i]);
                            }
                            ui.end_row();

                            for i in 0..3 {
                                ui.colored_label(species_colors[i], species_names[i]);
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
                    ui.add_space(5.0);
                    ui.vertical_centered(|ui| {
                        if ui.button("Reset Matrix").clicked() {
                            config.interaction_matrix = DEFAULT_INTERACTION_MATRIX;
                        }
                    });
                });

                ui.add_space(20.0);
                ui.separator();
                ui.add_space(10.0);

                ui.vertical_centered(|ui| {
                    if ui.button("🎲 I'm feeling lucky").on_hover_text("Randomize agent behavior and environment settings.").clicked() {
                        let mut rng = rand::rng();
                        config.sensor_angle = rng.random_range(0.0..180.0f32).to_radians();
                        config.sensor_dist = rng.random_range(1.0..50.0);
                        config.turn_speed = rng.random_range(50.0..2000.0f32).to_radians();
                        config.move_speed = rng.random_range(20.0..300.0);
                        config.jitter_amount = rng.random_range(0.0..0.5);
                        config.decay = 1.0 / rng.random_range(0.5..5.0); // evap time 0.5s to 5s
                        config.deposit_amount = rng.random_range(0.001..0.02);
                        config.diffuse_speed = rng.random_range(10.0..100.0);
                    }

                    ui.add_space(5.0);

                    if ui.button("Reset All to Defaults").clicked() {
                        config.sensor_angle = DEFAULT_SENSOR_ANGLE;
                        config.sensor_dist = DEFAULT_SENSOR_DIST;
                        config.turn_speed = DEFAULT_TURN_SPEED;
                        config.move_speed = DEFAULT_MOVE_SPEED;
                        config.decay = DEFAULT_DECAY;
                        config.active_agents = DEFAULT_AGENT_COUNT;
                        config.deposit_amount = DEFAULT_DEPOSIT_AMOUNT;
                        config.spawn_radius = DEFAULT_SPAWN_RADIUS;
                        config.jitter_amount = DEFAULT_JITTER_AMOUNT;
                        config.diffuse_speed = DEFAULT_DIFFUSE_SPEED;
                        config.species_weights = DEFAULT_SPECIES_WEIGHTS;
                        config.interaction_matrix = DEFAULT_INTERACTION_MATRIX;
                    }
                });

                ui.add_space(20.0);
            });

            ui.separator();
            ui.vertical_centered(|ui| {
                ui.add_space(5.0);
                ui.label(egui::RichText::new("Physarum Toy v0.1.0").weak().small());
                ui.hyperlink_to(
                    egui::RichText::new("GitHub Repository").weak().small(),
                    "https://github.com/danielmshiva/Physarum-Toy",
                );
            });
        });
}
