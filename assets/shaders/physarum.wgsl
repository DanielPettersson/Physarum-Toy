struct Amoeba {
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
var<storage, read_write> amoebas: array<Amoeba>;

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

fn sense(amoeba: Amoeba, sensor_angle_offset: f32) -> f32 {
    let sensor_angle = amoeba.angle + sensor_angle_offset;
    let sensor_dir = vec2<f32>(cos(sensor_angle), sin(sensor_angle));
    let sensor_pos = amoeba.pos + sensor_dir * config.sensor_dist;
    
    let x = (i32(sensor_pos.x) % i32(config.width) + i32(config.width)) % i32(config.width);
    let y = (i32(sensor_pos.y) % i32(config.height) + i32(config.height)) % i32(config.height);
    
    return textureLoad(trail_map, vec2<i32>(x, y)).r;
}

@compute @workgroup_size(64)
fn simulate(@builtin(global_invocation_id) id: vec3<u32>) {
    let amoeba_index = id.x;
    if (amoeba_index >= arrayLength(&amoebas)) {
        return;
    }
    
    var amoeba = amoebas[amoeba_index];
    
    // Sense pheromones
    let v_fwd = sense(amoeba, 0.0);
    let v_left = sense(amoeba, config.sensor_angle);
    let v_right = sense(amoeba, -config.sensor_angle);
        
    if (v_fwd > v_left && v_fwd > v_right) {
        // Continue forward
    } else if (v_fwd < v_left && v_fwd < v_right) {
        let random_val = f32(hash(amoeba_index ^ u32(amoeba.pos.x * 1000.0) ^ u32(amoeba.pos.y * 1000.0))) / 4294967295.0;
        amoeba.angle += (random_val - 0.5) * 2.0 * config.turn_speed * config.delta_time;
    } else if (v_left > v_right) {
        amoeba.angle += config.turn_speed * config.delta_time;
    } else if (v_right > v_left) {
        amoeba.angle -= config.turn_speed * config.delta_time;
    }
    
    // Move
    let dir = vec2<f32>(cos(amoeba.angle), sin(amoeba.angle));
    amoeba.pos += dir * config.move_speed * config.delta_time;
    
    // Wrap around boundaries
    amoeba.pos.x = fract(amoeba.pos.x / f32(config.width)) * f32(config.width);
    amoeba.pos.y = fract(amoeba.pos.y / f32(config.height)) * f32(config.height);
    
    amoebas[amoeba_index] = amoeba;
    
    // Deposit trail
    let x = i32(amoeba.pos.x);
    let y = i32(amoeba.pos.y);
    textureStore(trail_map, vec2<i32>(x, y), vec4<f32>(1.0, 1.0, 1.0, 1.0));
}

@compute @workgroup_size(16, 16)
fn diffuse(@builtin(global_invocation_id) id: vec3<u32>) {
    let x = i32(id.x);
    let y = i32(id.y);
    
    if (x >= i32(config.width) || y >= i32(config.height)) {
        return;
    }
    
    var sum = vec4<f32>(0.0);
    for (var i = -1; i <= 1; i++) {
        for (var j = -1; j <= 1; j++) {
            let nx = (x + i + i32(config.width)) % i32(config.width);
            let ny = (y + j + i32(config.height)) % i32(config.height);
            sum += textureLoad(trail_map, vec2<i32>(nx, ny));
        }
    }
    
    let blurred = sum / 9.0;
    let decay_factor = (1.0 - config.decay * config.delta_time);
    var decayed = vec4<f32>(blurred.rgb * decay_factor, 1.0);
        
    textureStore(trail_map, vec2<i32>(x, y), decayed);
}
