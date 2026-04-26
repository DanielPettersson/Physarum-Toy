# Physarum-Toy

A GPU-accelerated simulation of *Physarum polycephalum* (slime mold) built with **Bevy** and **WGSL** compute shaders.

## Overview

This project implements the agent-based model described by Jeff Jones in ["Characteristics of Pattern Formation and Evolution in Self-Organizing Soft-Iterative Systems"](https://uwe-repository.worktribe.com/output/980579). 

The simulation supports up to **2,000,000 individual agents** across multiple species that interact with each other through a configurable pheromone system.

## Features

- **High Performance**: Leverages WGSL compute shaders to simulate millions of agents at high frame rates.
- **Multiple Species**: Support for up to 3 distinct species (Red, Green, Blue) with unique pheromone trails.
- **Cross-Species Interactions**: Configurable attraction and repulsion weights via a real-time interaction matrix.
- **Dynamic Distribution**: Adjust the relative population weights of each species and respawn them instantly.
- **Interactive UI**: Real-time configuration of simulation parameters using `egui`.
- **Modern Tech Stack**: Built with Bevy 0.18, utilizing the latest render graph patterns and buffered message systems.

## Simulation Logic

Each agent follows a three-stage cycle:
1.  **Sense**: Detects pheromone intensity using three sensors (forward, left, right). The sensed value is a weighted dot product of the trail color and the species' unique interaction weights.
2.  **Turn & Move**: Rotates towards higher pheromone concentrations and moves forward at a speed scaled by the sensed intensity.
3.  **Deposit**: Leaves a pheromone trail in its species' specific color channel (Red, Green, or Blue).
4.  **Diffuse & Decay**: The global trail map undergoes Box Blur diffusion and linear decay every frame to simulate evaporation and spread.

## Requirements

- Rust (2024 edition)
- A GPU supporting Vulkan, Metal, or DX12 (WebGPU/WGPU compatible)

## Running the Simulation

To run the project with optimizations enabled:

```bash
cargo run --release
```

## Configuration (UI)

The simulation provides an interactive side panel where you can adjust:
- **Movement**: Sensor angle, distance, turn speed, and max move speed.
- **Environment**: Evaporation time and diffusion speed.
- **Species Distribution**: Relative spawning weights for Red, Green, and Blue populations.
- **Interaction Matrix**: Fine-grained control over how each species reacts to its own and others' pheromones (Attraction vs. Repulsion).

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
