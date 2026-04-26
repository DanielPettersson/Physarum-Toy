# Physarum-Toy

A GPU-accelerated simulation of *Physarum polycephalum* (slime mold) built with **Bevy** and **WGSL** compute shaders.

![Physarum Simulation](https://user-images.githubusercontent.com/1234567/physarum-demo.png) *(Note: Placeholder image link)*

## Overview

This project implements the agent-based model described by Jeff Jones in ["Characteristics of Pattern Formation and Evolution in Self-Organizing Soft-Iterative Systems"](https://uwe-repository.worktribe.com/output/980579). 

The simulation consists of 1,000,000 individual agents (agents) that:
1.  **Sense**: Detect pheromone intensity in front and to the sides using sensors.
2.  **Turn**: Rotate towards the direction with the highest pheromone concentration.
3.  **Move**: Advance forward at a speed scaled by the sensed pheromone intensity (20% to 100% of max speed) and deposit new pheromones onto a trail map.
4.  **Diffuse & Decay**: The trail map undergoes a box blur (diffusion) and constant evaporation (decay) every frame.

The emergent behavior produces complex, organic branching structures reminiscent of biological slime molds.

## Features

- **High Performance**: Leverages WGSL compute shaders to simulate 1 million agents at 60+ FPS.
- **Bevy Integration**: Built using the Bevy game engine's modern render graph and extraction patterns.
- **Real-time Visualization**: Direct rendering of the trail map texture using a Sprite.
- **FPS Overlay**: Built-in performance monitoring.

## Requirements

- Rust (2024 edition)
- A GPU supporting Vulkan, Metal, or DX12 (WGPU compatible)

## Running the Simulation

To run the project in release mode for best performance:

```bash
cargo run --release
```

## Configuration

You can tweak the simulation parameters in `src/main.rs`:

```rust
PhysarumConfig {
    sensor_angle: 0.35,  // Angle of sensors in radians
    sensor_dist: 15.0,   // Distance of sensors from agent
    turn_speed: 10.0,    // Turning rate
    move_speed: 50.0,    // Maximum movement speed (scales with trail intensity)
    decay: 1.0,          // Trail evaporation rate
    // ...
}
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
