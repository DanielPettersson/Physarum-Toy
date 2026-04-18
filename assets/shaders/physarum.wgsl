// Physarum simulation compute shader

struct Agent {
    pos: vec2<f32>,
    angle: f32,
    _pad: f32,
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
}

@group(0) @binding(0)
var<storage, read_write> agents: array<Agent>;

@group(0) @binding(1)
var trail_map: texture_storage_2d<rgba16float, read_write>;

@group(0) @binding(2)
var<uniform> config: Config;

// Simple pseudo-random function
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
    
    return textureLoad(trail_map, vec2<i32>(x, y)).r;
}

/// Agent simulation step: Senses pheromones, turns, moves, and deposits trail.
@compute @workgroup_size(64)
fn simulate(@builtin(global_invocation_id) id: vec3<u32>) {
    let agent_index = id.x;
    if (agent_index >= arrayLength(&agents)) {
        return;
    }
    
    var agent = agents[agent_index];
    
    // Sense pheromones in three directions: forward, left, and right
    let v_fwd = sense(agent, 0.0);
    let v_left = sense(agent, config.sensor_angle);
    let v_right = sense(agent, -config.sensor_angle);
        
    if (v_fwd > v_left && v_fwd > v_right) {
        // Continue forward if strongest trail is ahead
    } else if (v_fwd < v_left && v_fwd < v_right) {
        // Turn randomly if forward is weaker than both sides
        let random_val = f32(hash(agent_index ^ u32(agent.pos.x * 1000.0) ^ u32(agent.pos.y * 1000.0))) / 4294967295.0;
        agent.angle += (random_val - 0.5) * 2.0 * config.turn_speed * config.delta_time;
    } else if (v_left > v_right) {
        // Turn left if strongest trail is to the left
        agent.angle += config.turn_speed * config.delta_time;
    } else if (v_right > v_left) {
        // Turn right if strongest trail is to the right
        agent.angle -= config.turn_speed * config.delta_time;
    }
    
    // Move agent forward
    let dir = vec2<f32>(cos(agent.angle), sin(agent.angle));
    agent.pos += dir * config.move_speed * config.delta_time;
    
    // Wrap around boundaries
    agent.pos.x = fract(agent.pos.x / f32(config.width)) * f32(config.width);
    agent.pos.y = fract(agent.pos.y / f32(config.height)) * f32(config.height);
    
    agents[agent_index] = agent;
    
    // Deposit trail at new position
    let x = i32(agent.pos.x);
    let y = i32(agent.pos.y);
    textureStore(trail_map, vec2<i32>(x, y), vec4<f32>(1.0, 1.0, 1.0, 1.0));
}

/// Pheromone diffusion and decay step: Spreads trails and reduces intensity over time.
@compute @workgroup_size(16, 16)
fn diffuse(@builtin(global_invocation_id) id: vec3<u32>) {
    let x = i32(id.x);
    let y = i32(id.y);
    
    if (x >= i32(config.width) || y >= i32(config.height)) {
        return;
    }
    
    // Average values in 3x3 neighborhood (Box Blur)
    var sum = vec4<f32>(0.0);
    for (var i = -1; i <= 1; i++) {
        for (var j = -1; j <= 1; j++) {
            let nx = (x + i + i32(config.width)) % i32(config.width);
            let ny = (y + j + i32(config.height)) % i32(config.height);
            sum += textureLoad(trail_map, vec2<i32>(nx, ny));
        }
    }
    
    let blurred = sum / 9.0;
    
    // Apply decay
    let decay_factor = (1.0 - config.decay * config.delta_time);
    var decayed = vec4<f32>(blurred.rgb * decay_factor, 1.0);
        
    textureStore(trail_map, vec2<i32>(x, y), decayed);
}
