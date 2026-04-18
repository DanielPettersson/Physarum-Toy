#import bevy_ui::ui_vertex_output::UiVertexOutput
#import bevy_render::globals::Globals

@group(0) @binding(1)
var<uniform> globals: Globals;

struct CustomMaterial {
    color: vec4<f32>,
}

@group(1) @binding(0)
var<uniform> material: CustomMaterial;

@fragment
fn fragment(in: UiVertexOutput) -> @location(0) vec4<f32> {
    // Coordinate goes from 0 to 1
    let uv = in.uv;
    
    // Create an animated background
    let time = globals.time;
    let color = 0.5 + 0.5 * cos(time + uv.xyx + vec3<f32>(0.0, 2.0, 4.0));
    
    return vec4<f32>(color * material.color.rgb, 1.0);
}
