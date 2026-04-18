use bevy::{
    prelude::*,
    reflect::TypePath,
    render::render_resource::AsBindGroup,
};

fn main() {
    App::new()
        .add_plugins((
            DefaultPlugins,
            UiMaterialPlugin::<CustomMaterial>::default(),
        ))
        .add_systems(Startup, setup)
        .run();
}

fn setup(
    mut commands: Commands,
    mut ui_materials: ResMut<Assets<CustomMaterial>>,
) {
    // Camera is required to render anything
    commands.spawn(Camera2d);

    // Spawn a full-screen UI node with our custom material
    commands.spawn((
        MaterialNode(ui_materials.add(CustomMaterial {
            color: LinearRgba::WHITE,
        })),
        Node {
            width: Val::Percent(100.0),
            height: Val::Percent(100.0),
            ..default()
        },
    ));
}

#[derive(Asset, TypePath, AsBindGroup, Debug, Clone)]
struct CustomMaterial {
    #[uniform(0)]
    color: LinearRgba,
}

impl UiMaterial for CustomMaterial {
    fn fragment_shader() -> bevy::shader::ShaderRef {
        "shaders/fullscreen.wgsl".into()
    }
}
