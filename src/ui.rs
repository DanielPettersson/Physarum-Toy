use bevy::prelude::*;
use bevy_egui::{egui, EguiContexts};
use crate::simulation::*;

/// System that draws the configuration UI.
pub fn physarum_ui(
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
