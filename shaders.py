VERTEX_SHADER = """
#version 330

in vec3 in_position;
in vec3 in_normal;
in float in_material;

out vec3 v_normal;
out vec3 v_pos;
flat out float v_material;

uniform mat4 u_proj;
uniform mat4 u_view;

void main() {
    v_normal = in_normal;
    v_pos = in_position;
    v_material = in_material;
    gl_Position = u_proj * u_view * vec4(in_position, 1.0);
}
"""

FRAGMENT_SHADER = """
#version 330

in vec3 v_normal;
in vec3 v_pos;
flat in float v_material;

out vec4 fragColor;

void main() {
    int mat_id = int(v_material);
    
    if (mat_id == 0) {
        discard;
    }
    
    vec3 norm = normalize(v_normal);
    vec3 light_dir = normalize(vec3(0.4, 0.6, 1.0));
    float diff = max(abs(dot(norm, light_dir)), 0.4);

    vec3 base_color;
    float alpha = 1.0;
    
    if (mat_id == 1) {
        base_color = vec3(0.55, 0.53, 0.50); // Rock (Grey)
    } else if (mat_id == 2) {
        base_color = vec3(0.86, 0.80, 0.60); // Sand (Beige)
    } else if (mat_id == 3) {
        base_color = vec3(0.35, 0.65, 0.3); // Grass (Green)
    } else if (mat_id == 4) {
        base_color = vec3(0.2, 0.2, 0.9); // Water (Blue)
        alpha = 0.5;
    } else {
        base_color = vec3(1, 0, 1); // Null (Purple)
    }

    fragColor = vec4(base_color * diff, alpha);
}
"""

UI_VERTEX_SHADERS = {
'RECT_VS': """
#version 330
in vec2 in_position;

void main() {
    gl_Position = vec4(in_position, 0.0, 1.0);
}
""",

'TEXT_VS': """
#version 330
in vec2 in_position;
in vec2 in_texcoord;
out vec2 v_texcoord;

void main() {
    v_texcoord = in_texcoord;
    gl_Position = vec4(in_position, 0.0, 1.0);
}
"""
}

UI_FRAGMENT_SHADERS = {
'RECT_FS': """
#version 330
uniform vec3 u_color;
out vec4 fragColor;

void main() {
    fragColor = vec4(u_color, 1.0);
}
""",

'TEXT_FS': """
#version 330
uniform sampler2D u_texture;
in vec2 v_texcoord;
out vec4 fragColor;

void main() {
    fragColor = texture(u_texture, v_texcoord);
}
"""
}