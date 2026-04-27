// Physarum simulation compute shader

struct Agent {
    pos: vec2<f32>,
    angle: f32,
    species: u32,
}

struct Config {
    sensor_angle: f32,
    sensor_dist: f32,
    turn_speed: f32,
    move_speed: f32,
    decay: f32,
    width: u32,
    height: u32,
    delta_time: f32,
    diffuse_speed: f32,
    active_agents: u32,
    deposit_amount: f32,
    spawn_radius: f32,
    jitter_amount: f32,
    _padding1: f32,
    _padding2: f32,
    _padding3: f32,
    species_weights: vec4<f32>,
    interaction_matrix: mat4x4<f32>,
}

@group(0) @binding(0)
var<storage, read_write> agents: array<Agent>;

@group(0) @binding(1)
var trail_map: texture_storage_2d<rgba16float, read_write>;

@group(0) @binding(2)
var<uniform> config: Config;

@group(0) @binding(3)
var trail_map_temp: texture_storage_2d<rgba16float, read_write>;

// Simple pseudo-random function for agent behavior
fn hash(u: u32) -> u32 {
    var x = u;
    x = x ^ (x >> 16u);
    x = x * 0x85ebca6bu;
    x = x ^ (x >> 13u);
    x = x * 0xc2b2ae35u;
    x = x ^ (x >> 16u);
    return x;
}

// Senses the pheromone level at a specific angle relative to the agent's current orientation
fn sense(agent: Agent, sensor_angle_offset: f32) -> f32 {
    let sensor_angle = agent.angle + sensor_angle_offset;
    let sensor_dir = vec2<f32>(cos(sensor_angle), sin(sensor_angle));
    let sensor_pos = agent.pos + sensor_dir * config.sensor_dist;
    
    // Boundary wrap for sensing
    let x = (i32(sensor_pos.x) % i32(config.width) + i32(config.width)) % i32(config.width);
    let y = (i32(sensor_pos.y) % i32(config.height) + i32(config.height)) % i32(config.height);
    
    let trail = textureLoad(trail_map, vec2<i32>(x, y));
    return dot(trail, config.interaction_matrix[agent.species]);
}

/// Agent simulation step: Senses pheromones, turns, moves, and deposits trail.
@compute @workgroup_size(64)
fn simulate(@builtin(global_invocation_id) id: vec3<u32>) {
    let agent_index = id.x;
    if (agent_index >= config.active_agents) {
        return;
    }
    
    var agent = agents[agent_index];
    
    // Sense pheromones in 5 directions for a wider field of view
    let v_fwd = sense(agent, 0.0);
    let v_left = sense(agent, config.sensor_angle);
    let v_right = sense(agent, -config.sensor_angle);
    let v_far_left = sense(agent, config.sensor_angle * 2.0);
    let v_far_right = sense(agent, -config.sensor_angle * 2.0);
        
    // Calculate total weight of positive (attractive) signals
    let w_fwd = max(0.0, v_fwd);
    let w_l = max(0.0, v_left);
    let w_r = max(0.0, v_right);
    let w_fl = max(0.0, v_far_left);
    let w_fr = max(0.0, v_far_right);
    let total_weight = w_fwd + w_l + w_r + w_fl + w_fr;
    
    let random_val = f32(hash(agent_index ^ u32(agent.pos.x * 1000.0) ^ u32(agent.pos.y * 1000.0))) / 4294967295.0;

    // Use a small threshold to allow "sprouting" (ignoring very weak signals)
    if (total_weight > 0.05) {
        // Attraction: Continuous weighted steering towards the strongest signal
        let desired_offset = (
            config.sensor_angle * w_l + 
            -config.sensor_angle * w_r + 
            config.sensor_angle * 2.0 * w_fl + 
            -config.sensor_angle * 2.0 * w_fr
        ) / total_weight;
        
        // Add random jitter even when following a trail to encourage branching
        let jitter = (random_val - 0.5) * 2.0 * config.jitter_amount; 
        agent.angle += (desired_offset + jitter) * config.turn_speed * config.delta_time;
    } else if (v_fwd < 0.0 || v_left < 0.0 || v_right < 0.0 || v_far_left < 0.0 || v_far_right < 0.0) {
        // Repulsion: Everything seen is negative or neutral. Steer away from strongest repulsion.
        var min_val = v_fwd;
        var min_offset = 0.0;
        if (v_left < min_val) { min_val = v_left; min_offset = config.sensor_angle; }
        if (v_right < min_val) { min_val = v_right; min_offset = -config.sensor_angle; }
        if (v_far_left < min_val) { min_val = v_far_left; min_offset = config.sensor_angle * 2.0; }
        if (v_far_right < min_val) { min_val = v_far_right; min_offset = -config.sensor_angle * 2.0; }
        
        if (min_offset == 0.0) {
            // Most repulsive is forward: turn left or right to escape
            if ((hash(agent_index) % 2u) == 0u) {
                agent.angle += config.turn_speed * config.delta_time;
            } else {
                agent.angle -= config.turn_speed * config.delta_time;
            }
        } else {
            // Steer away from the repulsive source
            agent.angle -= sign(min_offset) * config.turn_speed * config.delta_time;
        }
    } else {
        // Exploration: No signal or below threshold. Brownian motion to find trails.
        agent.angle += (random_val - 0.5) * 2.0 * config.turn_speed * sqrt(max(config.delta_time, 0.0001) / 60.0);
    }
    
    // Scale movement speed based on sensed pheromones
    let max_sense = max(v_fwd, max(max(v_left, v_right), max(v_far_left, v_far_right)));
    let speed_multiplier = mix(0.2, 1.0, clamp(max_sense, 0.0, 1.0));
    
    // Move agent forward
    let dir = vec2<f32>(cos(agent.angle), sin(agent.angle));
    agent.pos += dir * config.move_speed * speed_multiplier * config.delta_time;
    
    // Wrap around boundaries
    agent.pos.x = fract(agent.pos.x / f32(config.width)) * f32(config.width);
    agent.pos.y = fract(agent.pos.y / f32(config.height)) * f32(config.height);
    
    agents[agent_index] = agent;
    
    // Deposit trail at new position
    let x = i32(agent.pos.x);
    let y = i32(agent.pos.y);
    
    // Select color channel based on species and use deposit_amount
    var deposit = vec4<f32>(0.0, 0.0, 0.0, 0.0);
    if (agent.species == 0u) {
        deposit.r = config.deposit_amount;
    } else if (agent.species == 1u) {
        deposit.g = config.deposit_amount;
    } else if (agent.species == 2u) {
        deposit.b = config.deposit_amount;
    }
    
    // Read current value and add new deposit, clamping to 1.0
    let current = textureLoad(trail_map, vec2<i32>(x, y));
    let new_val = min(current + deposit, vec4<f32>(1.0));
    
    textureStore(trail_map, vec2<i32>(x, y), new_val);
}

/// Horizontal blur pass
@compute @workgroup_size(16, 16)
fn diffuse_h(@builtin(global_invocation_id) id: vec3<u32>) {
    let x = i32(id.x);
    let y = i32(id.y);
    
    if (x >= i32(config.width) || y >= i32(config.height)) {
        return;
    }
    
    var sum = vec4<f32>(0.0);
    for (var i = -1; i <= 1; i++) {
        let nx = (x + i + i32(config.width)) % i32(config.width);
        sum += textureLoad(trail_map, vec2<i32>(nx, y));
    }
    
    textureStore(trail_map_temp, vec2<i32>(x, y), sum / 3.0);
}

/// Vertical blur pass with mixing and decay
@compute @workgroup_size(16, 16)
fn diffuse_v(@builtin(global_invocation_id) id: vec3<u32>) {
    let x = i32(id.x);
    let y = i32(id.y);
    
    if (x >= i32(config.width) || y >= i32(config.height)) {
        return;
    }
    
    var sum = vec4<f32>(0.0);
    for (var i = -1; i <= 1; i++) {
        let ny = (y + i + i32(config.height)) % i32(config.height);
        sum += textureLoad(trail_map_temp, vec2<i32>(x, ny));
    }
    
    let blurred = sum / 3.0;
    
    // Time-dependent diffusion: Mix original pixel with blurred pixel based on speed and delta_time
    let original = textureLoad(trail_map, vec2<i32>(x, y));
    let mix_factor = clamp(config.diffuse_speed * config.delta_time, 0.0, 1.0);
    let diffused = mix(original, blurred, mix_factor);

    // Apply decay (clamped to avoid negative colors at very low framerates)
    let decay_factor = max(0.0, 1.0 - config.decay * config.delta_time);
    var decayed = vec4<f32>(diffused.rgb * decay_factor, 1.0);
        
    textureStore(trail_map, vec2<i32>(x, y), decayed);
}
