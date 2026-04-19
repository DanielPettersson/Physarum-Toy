// Physarum simulation compute shader with food, health, and mitosis
// Separated logic and display for maximum stability

struct Agent {
    pos: vec2<f32>,
    angle: f32,
    health: f32,
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
    food_attraction: f32,
    food_depletion_rate: f32,
    health_decay_rate: f32,
    food_health_gain: f32,
    show_food: f32,
    _pad1: f32,
}

struct SpawnData {
    count: atomic<u32>,
    _pad1: u32,
    _pad2: u32,
    _pad3: u32,
    indices: array<u32>,
}

@group(0) @binding(0)
var<storage, read_write> agents: array<Agent>;

@group(0) @binding(1)
var trail_map: texture_storage_2d<rgba16float, read_write>; // Pure pheromone map

@group(0) @binding(2)
var food_map: texture_storage_2d<r32float, read_write>; // Pure food map

@group(0) @binding(3)
var<storage, read_write> spawn_data: SpawnData;

@group(0) @binding(4)
var display_map: texture_storage_2d<rgba8unorm, read_write>; // Composite for display only

@group(0) @binding(5)
var<uniform> config: Config;

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

// Senses combined pheromone and food levels
fn sense(pos: vec2<f32>, sensor_angle: f32) -> f32 {
    let sensor_dir = vec2<f32>(cos(sensor_angle), sin(sensor_angle));
    let sensor_pos = pos + sensor_dir * config.sensor_dist;
    
    // Boundary wrap for sensing
    let x = (i32(sensor_pos.x) % i32(config.width) + i32(config.width)) % i32(config.width);
    let y = (i32(sensor_pos.y) % i32(config.height) + i32(config.height)) % i32(config.height);
    
    let pheromone = textureLoad(trail_map, vec2<i32>(x, y)).r;
    let food = textureLoad(food_map, vec2<i32>(x, y)).r;
    
    return pheromone + food * config.food_attraction;
}

/// Agent simulation step: Senses pheromones/food, turns, moves, eats, and manages lifecycle.
@compute @workgroup_size(64)
fn simulate(@builtin(global_invocation_id) id: vec3<u32>) {
    let agent_index = id.x;
    var agent = agents[agent_index];
    if (agent.health <= 0.0) {
        return;
    }
    
    // Sense pheromones and food in three directions: forward, left, and right
    let v_fwd = sense(agent.pos, agent.angle);
    let v_left = sense(agent.pos, agent.angle + config.sensor_angle);
    let v_right = sense(agent.pos, agent.angle - config.sensor_angle);
        
    if (v_fwd > v_left && v_fwd > v_right) {
        // Continue forward if strongest trail is ahead
    } else if (v_fwd < v_left && v_fwd < v_right) {
        // Turn randomly if forward is weaker than both sides.
        let random_val = f32(hash(agent_index ^ u32(agent.pos.x * 1000.0) ^ u32(agent.pos.y * 1000.0))) / 4294967295.0;
        agent.angle += (random_val - 0.5) * 2.0 * config.turn_speed * sqrt(max(config.delta_time, 0.0001) / 60.0);
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
    
    // Eating and health mechanics
    let ix = i32(agent.pos.x);
    let iy = i32(agent.pos.y);
    let food_val = textureLoad(food_map, vec2<i32>(ix, iy)).r;
    
    if (food_val > 0.0) {
        let consume = min(food_val, config.food_depletion_rate * config.delta_time);
        textureStore(food_map, vec2<i32>(ix, iy), vec4<f32>(food_val - consume, 0.0, 0.0, 0.0));
        agent.health += consume * config.food_health_gain;
    } else {
        agent.health -= config.health_decay_rate * config.delta_time;
    }
    
    // Lifecycle: Mitosis and Death
    if (agent.health <= 0.0) {
        // Death: Add index to dead list
        agent.health = 0.0;
        let old_count = atomicAdd(&spawn_data.count, 1u);
        if (old_count < arrayLength(&spawn_data.indices)) {
            spawn_data.indices[old_count] = agent_index;
        }
    } else if (agent.health >= 1.0) {
        // Mitosis: Try to spawn a child
        var popped = false;
        var child_index = 0u;
        loop {
            let count = atomicLoad(&spawn_data.count);
            if (count == 0u) { break; }
            let res = atomicCompareExchangeWeak(&spawn_data.count, count, count - 1u);
            if (res.exchanged) {
                child_index = spawn_data.indices[count - 1u];
                popped = true;
                break;
            }
        }
        
        if (popped) {
            agent.health = 0.5;
            var child = agent;
            child.angle += 3.14159 * 0.5; 
            agents[child_index] = child;
        }
    }
    
    agents[agent_index] = agent;
    
    // Deposit trail if alive, proportional to health
    if (agent.health > 0.0) {
        textureStore(trail_map, vec2<i32>(ix, iy), vec4<f32>(agent.health, 0.0, 0.0, 1.0));
    }
}

/// Pheromone diffusion and display compositing.
@compute @workgroup_size(16, 16)
fn diffuse(@builtin(global_invocation_id) id: vec3<u32>) {
    let x = i32(id.x);
    let y = i32(id.y);
    
    if (x >= i32(config.width) || y >= i32(config.height)) {
        return;
    }
    
    // Blur pure pheromone (R channel)
    var sum = 0.0;
    for (var i = -1; i <= 1; i++) {
        for (var j = -1; j <= 1; j++) {
            let nx = (x + i + i32(config.width)) % i32(config.width);
            let ny = (y + j + i32(config.height)) % i32(config.height);
            sum += textureLoad(trail_map, vec2<i32>(nx, ny)).r;
        }
    }
    
    let blurred = sum / 9.0;
    let original = textureLoad(trail_map, vec2<i32>(x, y)).r;
    let mix_factor = clamp(config.diffuse_speed * config.delta_time, 0.0, 1.0);
    let diffused = mix(original, blurred, mix_factor);

    // Apply decay to pure pheromone
    let decay_factor = max(0.0, 1.0 - config.decay * config.delta_time);
    let pheromone = diffused * decay_factor;
    
    // Write pure pheromone back to trail_map
    textureStore(trail_map, vec2<i32>(x, y), vec4<f32>(pheromone, 0.0, 0.0, 1.0));
    
    // Composite for display
    let food = textureLoad(food_map, vec2<i32>(x, y)).r * config.show_food;
    let final_color = vec4<f32>(pheromone + food, food, food, 1.0);
        
    textureStore(display_map, vec2<i32>(x, y), final_color);
}
