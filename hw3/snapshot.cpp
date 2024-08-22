// bump mapping + point_color + fog
#include <SDL2/SDL.h>
#include <GL/glew.h>

#define GLM_FORCE_SWIZZLE
#define GLM_ENABLE_EXPERIMENTAL
#include <glm/vec3.hpp>
#include <glm/mat4x4.hpp>
#include <glm/ext/matrix_transform.hpp>
#include <glm/ext/matrix_clip_space.hpp>
#include <glm/ext/scalar_constants.hpp>
#include <glm/gtx/string_cast.hpp>

#define TINYOBJLOADER_IMPLEMENTATION
#include "tiny_obj_loader.h"

#include "obj_parser.hpp"

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

#include <string_view>
#include <stdexcept>
#include <iostream>
#include <chrono>
#include <vector>
#include <cmath>
#include <cassert>
#include <map>
#include <numeric>
#include <random>
#include <algorithm>
#include <iterator>
#include <set>

std::random_device rd;
std::mt19937 g(rd());

std::string to_string(std::string_view str)
{
    return std::string(str.begin(), str.end());
}

void sdl2_fail(std::string_view message)
{
    throw std::runtime_error(to_string(message) + SDL_GetError());
}

void glew_fail(std::string_view message, GLenum error)
{
    throw std::runtime_error(to_string(message) + reinterpret_cast<const char *>(glewGetErrorString(error)));
}

const char shadow_vertex_shader_source[] =
R"(#version 330 core
uniform mat4 model;
uniform mat4 transform;

layout (location = 0) in vec3 in_position;
layout (location = 2) in vec2 in_texcoord;

out vec2 texcoord;

void main()
{
    gl_Position = transform * model * vec4(in_position, 1.0);
    texcoord = vec2(in_texcoord.x, 1.f - in_texcoord.y);
}
)";

const char shadow_fragment_shader_source[] =
R"(#version 330 core

in vec4 gl_FragCoord;
in vec2 texcoord;

uniform int flag_map_d;
uniform sampler2D alpha_texture;

layout (location = 0) out vec4 out_color;

void main()
{   
    if (flag_map_d == 1) {
        if (texture(alpha_texture, texcoord).r < 0.5) {
            discard;
        }
    }

    float z = gl_FragCoord.z;
    out_color = vec4(z, z * z + 0.25 * (dFdx(z)*dFdx(z) + dFdy(z)*dFdy(z)), 0.0, 0.0);
}
)";


const char vertex_shader_source[] =
R"(#version 330 core
uniform mat4 model;
uniform mat4 view;
uniform mat4 projection;

layout (location = 0) in vec3 in_position;
layout (location = 1) in vec3 in_normal;
layout (location = 2) in vec2 in_texcoord;

out vec3 position;
out vec3 normal;
out vec2 texcoord;

void main()
{
    gl_Position = projection * view * model * vec4(in_position, 1.0);
    position = (model * vec4(in_position, 1.0)).xyz;
    normal = normalize(mat3(model) * in_normal);
    texcoord = vec2(in_texcoord.x, 1.f - in_texcoord.y);
}
)";

const char fragment_shader_source[] =
R"(#version 330 core

uniform vec3 ambient;
uniform vec3 light_direction;
uniform vec3 light_color;
uniform vec3 camera_position;

uniform mat4 transform;
uniform sampler2D shadow_map;
uniform sampler2D ambient_texture;

uniform float glossiness;
uniform float power;

uniform sampler2D alpha_texture;
uniform int flag_map_d;

uniform sampler2D bump_texture;
uniform int flag_map_bump;

uniform sampler2D specular_texture;
uniform int flag_map_Ks;

uniform vec3 point_light_position;
uniform vec3 point_light_color;
uniform vec3 point_light_attenuation;

uniform samplerCube depthMap;
uniform float far_plane;

in vec3 position;
in vec3 normal;
in vec2 texcoord;

layout (location = 0) out vec4 out_color;

vec3 bbox_min = vec3(-2.f, -1.f, -1.f);
vec3 bbox_max = vec3(2.f,  1.f,  1.f);

float diffuse(vec3 direction, vec3 inner_normal) {
    return max(0.0, dot(inner_normal, direction));
}

float specular(vec3 direction, vec3 inner_normal) {
    vec3 reflected_direction = 2.0 * inner_normal * dot(inner_normal, direction) - direction;
    vec3 view_direction = normalize(camera_position - position);
    float factor = 1.0;
    if (flag_map_Ks == 1) {
        factor = texture(specular_texture, texcoord).r;
    }
    return factor * glossiness * pow(max(0.0, dot(reflected_direction, view_direction)), power);
}

float phong(vec3 direction, vec3 inner_normal) {
    return diffuse(direction, inner_normal) + specular(direction, inner_normal);
}

float ShadowCalculation() {
    vec3 fragToLight = point_light_position - position;
    float closestDepth = texture(depthMap, fragToLight).r;
    closestDepth *= far_plane;
    float currentDepth = distance(position, point_light_position);
    float bias = 0.005;
    float shadow = currentDepth - bias > closestDepth ? 1.0 : 0.0;
    return shadow;
}

float calc_factor(vec2 data, float sh_z) {
    float mu = data.r;
    float sigma = data.g - mu * mu;
    float z = sh_z - 0.001;
    float factor = (z < mu) ? 1.0 : sigma / (sigma + (z - mu) * (z - mu));

    float delta = 0.125;
    if (factor < delta) {
        factor = 0;
    }
    else {
        factor = (factor - delta) / (1 - delta);
    }

    return factor;
}

// BUMP MAPPING
vec3 perturb_normal(vec3 inner_normal) {
    vec3 sigma_s = dFdx(position);
    vec3 sigma_t = dFdy(position);

    vec3 r1 = cross(sigma_t, inner_normal);
    vec3 r2 = cross(inner_normal, sigma_s);

    float det = dot(sigma_s, r1);

    vec2 tex_dx = dFdx(texcoord);
    vec2 tex_dy = dFdy(texcoord);
    vec2 tex_ll = texcoord;
    vec2 tex_lr = texcoord + tex_dx;
    vec2 tex_ul = texcoord + tex_dy;
    float height_ll = 3.f * texture(bump_texture, tex_ll).r;
    float height_lr = 3.f * texture(bump_texture, tex_lr).r;
    float height_ul = 3.f * texture(bump_texture, tex_ul).r;

    float dbs = height_lr - height_ll;
    float dbt = height_ul - height_ll;

    vec3 gradient = sign(det) * (dbs * r1 + dbt * r2);

    return normalize(abs(det) * inner_normal - gradient);
}

// TONE MAPPING
vec3 TonemapRaw(vec3 x)
{
    float A = 0.15;
    float B = 0.50;
    float C = 0.10;
    float D = 0.20;
    float E = 0.02;
    float F = 0.30;
    return ((x*(A*x+C*B)+D*E)/(x*(A*x+B)+D*F))-E/F;
}
vec3 Uncharted2Tonemap(vec3 color)
{
    float W = 11.2;
    return TonemapRaw(color) / TonemapRaw(vec3(W));
}

void main()
{
    if (flag_map_d == 1) {
        if (texture(alpha_texture, texcoord).r < 0.5) {
            discard;
        }
    }

    vec3 real_normal = normal;

    if (flag_map_bump == 1) {
        real_normal = perturb_normal(normal);
    }

    vec4 shadow_pos = transform * vec4(position, 1.0);
    shadow_pos /= shadow_pos.w;
    shadow_pos = shadow_pos * 0.5 + vec4(0.5);
    
    bool in_shadow_texture = (shadow_pos.x >= 0.0) && (shadow_pos.x < 1.0) && 
        (shadow_pos.y >= 0.0) && (shadow_pos.y < 1.0) && 
        (shadow_pos.z >= 0.0) && (shadow_pos.z < 1.0);

    float factor = 1.0;

    //code of gaussian blur below
    vec4 sum = vec4(0.0);
    float sum_w = 0.0;
    const int N = 4;
    float radius = 5.0;

    for (int x = -N; x <= N; ++x) {
        for (int y = -N; y <= N; ++y) {
            float c = exp(-float(x * x + y * y) / (radius * radius));
            sum += c * texture(shadow_map, shadow_pos.xy + vec2(x,y) / vec2(textureSize(shadow_map, 0)));
            sum_w += c;
        }
    }

    // code of VARIANCE SHADOW MAP (VSM)
    vec4 data = sum / sum_w;
    float bias = -0.005;
    float mu = data.r;
    float sigma = data.g - mu * mu;
    float z = shadow_pos.z + bias;
    factor = (z < mu) ? 1.0 : sigma * sigma / (sigma * sigma + (z - mu) * (z - mu));

    float delta = 0.125;
    if (factor < delta) {
        factor = 0;
    }
    else {
        factor = (factor - delta) / (1 - delta);
    }
    
    float shadow_factor = 1.0;
    if (in_shadow_texture) {
        shadow_factor = factor;
    }

    vec3 albedo = texture(ambient_texture, texcoord).rgb;

    float point_light_distance = distance(position, point_light_position);
    vec3 point_light_direction = normalize(point_light_position - position);
    vec3 point_color = phong(point_light_direction, real_normal) * point_light_color / (point_light_attenuation.x + point_light_attenuation.y * point_light_distance + point_light_attenuation.z * point_light_distance * point_light_distance);
    float point_shadow = ShadowCalculation();

    vec3 light = albedo * ambient;
    light += light_color * phong(light_direction, real_normal) * shadow_factor;
    light += point_color * point_shadow;
    vec3 color = light;
    out_color = vec4(color, 1.0);
    color = pow(Uncharted2Tonemap(color), vec3(1.0 / 2.2));
    out_color = vec4(color, 1.0);

    float fog_absorption = 0.0015;
    vec3 fog_color = vec3(221, 226, 227) / 255;

    float camera_distance = length(camera_position - position);
    float optical_depth = camera_distance * fog_absorption;
    float fog_factor = 1 - exp(-optical_depth);
    color = mix(color, fog_color, fog_factor);
    out_color = vec4(color, 1.0);

    // out_color = vec4(real_normal,1.0);
}
)";

const char point_shadow_vertex_shader_source[] =
R"(
#version 330 core
layout (location = 0) in vec3 in_position;

uniform mat4 model;

void main()
{
    gl_Position = model * vec4(in_position, 1.0);
}

)";

const char point_shadow_geometry_shader_source[] =
R"(
#version 330 core
layout (triangles) in;
layout (triangle_strip, max_vertices=18) out;

out vec4 FragPos;
uniform mat4 shadowMatrices[6];

void main() {
    for (int face = 0; face < 6; ++face) {
        gl_Layer = face;
        for (int i = 0; i < 3; ++i) {
            FragPos = gl_in[i].gl_Position;
            gl_Position = shadowMatrices[face] * FragPos;
            EmitVertex();
        }
        EndPrimitive();
    }
}

)";

const char point_shadow_fragment_shader_source[] =
R"(
#version 330 core

in vec4 FragPos;
uniform vec3 lightPos;
uniform float far_plane;

void main() {
    float lightDistance = length(FragPos.xyz-lightPos);
    lightDistance /= far_plane;
    gl_FragDepth = lightDistance;
}
)";


GLuint create_shader(GLenum type, const char * source)
{
    GLuint result = glCreateShader(type);
    glShaderSource(result, 1, &source, nullptr);
    glCompileShader(result);
    GLint status;
    glGetShaderiv(result, GL_COMPILE_STATUS, &status);
    if (status != GL_TRUE)
    {
        GLint info_log_length;
        glGetShaderiv(result, GL_INFO_LOG_LENGTH, &info_log_length);
        std::string info_log(info_log_length, '\0');
        glGetShaderInfoLog(result, info_log.size(), nullptr, info_log.data());
        throw std::runtime_error("Shader compilation failed: " + info_log);
    }
    return result;
}

GLuint create_program(GLuint vertex_shader, GLuint fragment_shader,
                      std::optional<GLuint> geometry_shader = std::nullopt) {
    GLuint result = glCreateProgram();
    glAttachShader(result, vertex_shader);
    glAttachShader(result, fragment_shader);
    if (geometry_shader.has_value()) {
        glAttachShader(result, *geometry_shader);
    }

    glLinkProgram(result);

    GLint status;
    glGetProgramiv(result, GL_LINK_STATUS, &status);
    if (status != GL_TRUE) {
        GLint info_log_length;
        glGetProgramiv(result, GL_INFO_LOG_LENGTH, &info_log_length);
        std::string info_log(info_log_length, '\0');
        glGetProgramInfoLog(result, info_log.size(), nullptr, info_log.data());
        throw std::runtime_error("Program linkage failed: " + info_log);
    }

  return result;
}

std::vector<obj_data::vertex> get_vertices(
        const tinyobj::attrib_t &attrib,
        const std::vector<tinyobj::shape_t> &shapes) {

    std::vector<obj_data::vertex> res;
    for(auto &shape : shapes) {
        for (auto &i: shape.mesh.indices) {
            res.push_back({{
                    attrib.vertices[3 * i.vertex_index],
                    attrib.vertices[3 * i.vertex_index + 1],
                    attrib.vertices[3 * i.vertex_index + 2]
                }, {
                    attrib.normals[3 * i.normal_index],
                    attrib.normals[3 * i.normal_index + 1],
                    attrib.normals[3 * i.normal_index + 2]
                }, {
                    attrib.texcoords[2 * i.texcoord_index],
                    attrib.texcoords[2 * i.texcoord_index + 1]
                }
            });
        }
    }
    return res;
}

typedef std::array<glm::vec3, 8> bounding_box;

bounding_box get_bounding_box(const std::vector<obj_data::vertex> &scene) {
    float x_bounds[2] = {std::numeric_limits<float>::infinity(),
                         -std::numeric_limits<float>::infinity()};
    float y_bounds[2] = {std::numeric_limits<float>::infinity(),
                         -std::numeric_limits<float>::infinity()};
    float z_bounds[2] = {std::numeric_limits<float>::infinity(),
                         -std::numeric_limits<float>::infinity()};
    for(auto &v : scene) {
        x_bounds[0] = std::min(x_bounds[0], v.position[0]);
        y_bounds[0] = std::min(y_bounds[0], v.position[1]);
        z_bounds[0] = std::min(z_bounds[0], v.position[2]);
        x_bounds[1] = std::max(x_bounds[1], v.position[0]);
        y_bounds[1] = std::max(y_bounds[1], v.position[1]);
        z_bounds[1] = std::max(z_bounds[1], v.position[2]);
    }
    bounding_box res;
    for(int i = 0; i < 2; i++)
        for(int j = 0; j < 2; j++)
            for(int k = 0; k < 2; k++)
                res[1 * i + 2 * j + 4 * k] =
                        glm::vec3(x_bounds[i], y_bounds[j], z_bounds[k]);
    return res;
}

class texture_holder {
public:
    explicit texture_holder(GLint first_unit);
    GLint load_texture(const std::string &path, bool flag);
    GLint get_texture(const std::string &path, bool flag);

private:
    GLint m_first_unit;
    std::unordered_map<std::string, std::pair<GLuint, GLint>> m_textures;
};

GLint texture_holder::load_texture(const std::string &path, bool flag = false) {
    GLint unit = m_first_unit + (GLint)m_textures.size();
    auto it = m_textures.find(path);
    if(it != m_textures.end()) return it->second.second;
    glGenTextures(1, &m_textures[path].first);
    m_textures[path].second = unit;
    glActiveTexture(GL_TEXTURE0 + unit);
    glBindTexture(GL_TEXTURE_2D, m_textures[path].first);
    glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR);
    glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    int x, y, channels_in_file;
    auto pixels = stbi_load(path.c_str(), &x, &y, &channels_in_file, 4);
    glTexImage2D(GL_TEXTURE_2D, 0, (flag ? GL_SRGB8_ALPHA8 : GL_RGBA8), x, y, 0, GL_RGBA, GL_UNSIGNED_BYTE, pixels);
    glGenerateMipmap(GL_TEXTURE_2D);
    stbi_image_free(pixels);
    return unit;
}

GLint texture_holder::get_texture(const std::string &path, bool flag = false) {
    auto it = m_textures.find(path);
    if (it == m_textures.end()) {
        return load_texture(path, flag);
    }
    auto unit = it->second.second;
    glActiveTexture(GL_TEXTURE0 + unit);
    return unit;
}

texture_holder::texture_holder(GLint first_unit) : m_first_unit(first_unit) {}

std::set<int> convertToSet(std::vector<int> v) {
    std::set<int> s;

    for (int x : v) {
        s.insert(x);
    }
 
    return s;
}

glm::mat4 rotation_matrix(glm::vec3 camera_rotation) {
    glm::mat4 view(1.f);
    view = glm::rotate(view, camera_rotation.x, {1, 0, 0});
    view = glm::rotate(view, camera_rotation.y, {0, 1, 0});
    view = glm::rotate(view, camera_rotation.z, {0, 0, 1});
    return view;
}

int main() try
{
    if (SDL_Init(SDL_INIT_VIDEO) != 0)
        sdl2_fail("SDL_Init: ");

    SDL_GL_SetAttribute(SDL_GL_CONTEXT_MAJOR_VERSION, 3);
    SDL_GL_SetAttribute(SDL_GL_CONTEXT_MINOR_VERSION, 3);
    SDL_GL_SetAttribute(SDL_GL_CONTEXT_PROFILE_MASK, SDL_GL_CONTEXT_PROFILE_CORE);
    SDL_GL_SetAttribute(SDL_GL_DOUBLEBUFFER, 1);
    SDL_GL_SetAttribute(SDL_GL_MULTISAMPLEBUFFERS, 1);
    SDL_GL_SetAttribute(SDL_GL_MULTISAMPLESAMPLES, 4);
    SDL_GL_SetAttribute(SDL_GL_RED_SIZE, 8);
    SDL_GL_SetAttribute(SDL_GL_GREEN_SIZE, 8);
    SDL_GL_SetAttribute(SDL_GL_BLUE_SIZE, 8);
    SDL_GL_SetAttribute(SDL_GL_DEPTH_SIZE, 24);

    SDL_Window * window = SDL_CreateWindow("Homework 3",
        SDL_WINDOWPOS_CENTERED,
        SDL_WINDOWPOS_CENTERED,
        800, 600,
        SDL_WINDOW_OPENGL | SDL_WINDOW_RESIZABLE | SDL_WINDOW_MAXIMIZED);

    if (!window)
        sdl2_fail("SDL_CreateWindow: ");

    int width, height;
    SDL_GetWindowSize(window, &width, &height);

    SDL_GLContext gl_context = SDL_GL_CreateContext(window);
    if (!gl_context)
        sdl2_fail("SDL_GL_CreateContext: ");

    if (auto result = glewInit(); result != GLEW_NO_ERROR)
        glew_fail("glewInit: ", result);

    if (!GLEW_VERSION_3_3)
        throw std::runtime_error("OpenGL 3.3 is not supported");

    glClearColor(0.8f, 0.8f, 1.f, 0.f);

    auto vertex_shader = create_shader(GL_VERTEX_SHADER, vertex_shader_source);
    auto fragment_shader = create_shader(GL_FRAGMENT_SHADER, fragment_shader_source);
    auto program = create_program(vertex_shader, fragment_shader);

    GLuint model_location = glGetUniformLocation(program, "model");
    GLuint view_location = glGetUniformLocation(program, "view");
    GLuint projection_location = glGetUniformLocation(program, "projection");
    GLuint transform_location = glGetUniformLocation(program, "transform");
    GLuint ambient_location = glGetUniformLocation(program, "ambient");
    GLuint light_direction_location = glGetUniformLocation(program, "light_direction");
    GLuint light_color_location = glGetUniformLocation(program, "light_color");
    GLuint camera_position_location = glGetUniformLocation(program, "camera_position");
    GLuint shadow_map_location = glGetUniformLocation(program, "shadow_map");
    GLuint ambient_texture_location = glGetUniformLocation(program, "ambient_texture");
    GLuint glossiness_location = glGetUniformLocation(program, "glossiness");
    GLuint power_location = glGetUniformLocation(program, "power");
    GLuint alpha_texture_location = glGetUniformLocation(program, "alpha_texture");
    GLuint flag_map_d_location = glGetUniformLocation(program, "flag_map_d");
    GLuint bump_texture_location = glGetUniformLocation(program, "bump_texture");
    GLuint flag_map_bump_location = glGetUniformLocation(program, "flag_map_bump");
    GLuint specular_texture_location = glGetUniformLocation(program, "specular_texture");
    GLuint flag_map_Ks_location = glGetUniformLocation(program, "flag_map_Ks");
    GLuint point_light_position_location = glGetUniformLocation(program, "point_light_position");
    GLuint point_light_color_location = glGetUniformLocation(program, "point_light_color");
    GLuint point_light_attenuation_location = glGetUniformLocation(program, "point_light_attenuation");
    GLuint depthMap_location = glGetUniformLocation(program, "depthMap");
    GLuint far_plane_location = glGetUniformLocation(program, "far_plane");


    auto shadow_vertex_shader = create_shader(GL_VERTEX_SHADER, shadow_vertex_shader_source);
    auto shadow_fragment_shader = create_shader(GL_FRAGMENT_SHADER, shadow_fragment_shader_source);
    auto shadow_program = create_program(shadow_vertex_shader, shadow_fragment_shader);

    GLuint shadow_model_location = glGetUniformLocation(shadow_program, "model");
    GLuint shadow_transform_location = glGetUniformLocation(shadow_program, "transform");
    GLuint shadow_alpha_texture_location = glGetUniformLocation(shadow_program, "alpha_texture");
    GLuint shadow_flag_map_d_location = glGetUniformLocation(shadow_program, "flag_map_d");

    /// *** Создаем шейдеры для тени от точечного источника
    auto point_shadow_vertex_shader = create_shader(GL_VERTEX_SHADER, point_shadow_vertex_shader_source);
    auto point_shadow_geometry_shader = create_shader(GL_GEOMETRY_SHADER, point_shadow_geometry_shader_source);
    auto point_shadow_fragment_shader = create_shader(GL_FRAGMENT_SHADER, point_shadow_fragment_shader_source);
    auto point_shadow_program = create_program(point_shadow_vertex_shader, point_shadow_fragment_shader, point_shadow_geometry_shader);

    GLuint point_shadow_model_location = glGetUniformLocation(point_shadow_program, "model");
    GLuint point_shadow_shadow_matrices_location = glGetUniformLocation(point_shadow_program, "shadowMatrices");
    GLuint point_shadow_light_pos_location = glGetUniformLocation(point_shadow_program, "lightPos");
    GLuint point_shadow_far_plane_location = glGetUniformLocation(point_shadow_program, "far_plane");

    std::string project_root = PROJECT_ROOT;
    std::string obj_path = project_root + "/sponza.obj";
    std::string materials_dir = project_root + "/";

    std::cout << "obj_path = " << obj_path << std::endl;
    std::cout << "materials_dir = " << materials_dir << std::endl;

    tinyobj::attrib_t attrib;
    std::vector<tinyobj::shape_t> shapes;
    std::vector<tinyobj::material_t> materials;

    tinyobj::LoadObj(&attrib, &shapes, &materials, nullptr, nullptr, obj_path.c_str(), materials_dir.c_str());

    std::cout << "after LoadObj" << std::endl;

    obj_data scene;
    // Loop over shapes
    for (size_t s = 0; s < shapes.size(); s++) {
        for (auto &i: shapes[s].mesh.indices) {
            scene.vertices.push_back({{
                    attrib.vertices[3 * i.vertex_index],
                    attrib.vertices[3 * i.vertex_index + 1],
                    attrib.vertices[3 * i.vertex_index + 2]
                }, {
                    attrib.normals[3 * i.normal_index],
                    attrib.normals[3 * i.normal_index + 1],
                    attrib.normals[3 * i.normal_index + 2]
                }, {
                    attrib.texcoords[2 * i.texcoord_index],
                    attrib.texcoords[2 * i.texcoord_index + 1]
                }
            });
        }
    }

    texture_holder textures(2);
    int counter = 0;

    for(auto &material : materials) {
        std::string texture_path = materials_dir + material.ambient_texname;
        std::replace(texture_path.begin(), texture_path.end(), '\\', '/');

        textures.load_texture(texture_path, true);
        std::cout << "material bump texture = " << material.bump_texname << std::endl << std::endl;
    }
    
    auto bounding_box = get_bounding_box(scene.vertices);
    glm::vec3 c = std::accumulate(bounding_box.begin(), bounding_box.end(), glm::vec3(0.f)) / 8.f;
    
    GLuint vao, vbo;
    glGenVertexArrays(1, &vao);
    glBindVertexArray(vao);

    glGenBuffers(1, &vbo);
    glBindBuffer(GL_ARRAY_BUFFER, vbo);
    glBufferData(GL_ARRAY_BUFFER, scene.vertices.size() * sizeof(obj_data::vertex), scene.vertices.data(), GL_STATIC_DRAW);

    glEnableVertexAttribArray(0);
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, sizeof(obj_data::vertex), (void*)(0));
    glEnableVertexAttribArray(1);
    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, sizeof(obj_data::vertex), (void*)(12));
    glEnableVertexAttribArray(2);
    glVertexAttribPointer(2, 2, GL_FLOAT, GL_FALSE, sizeof(obj_data::vertex), (void*)(24));

    GLsizei shadow_map_resolution = 2048;
    GLuint shadow_map;
    glGenTextures(1, &shadow_map);
    glActiveTexture(GL_TEXTURE0 + 1);
    glBindTexture(GL_TEXTURE_2D, shadow_map);
    glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RG32F, shadow_map_resolution, shadow_map_resolution, 0, GL_RGBA, GL_FLOAT, NULL);
    
    GLuint shadow_fbo;
    glGenFramebuffers(1, &shadow_fbo);
    glBindFramebuffer(GL_DRAW_FRAMEBUFFER, shadow_fbo);
    glFramebufferTexture(GL_DRAW_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, shadow_map, 0);
    
    GLuint rbo;
    glGenRenderbuffers(1, &rbo);
    glBindRenderbuffer(GL_RENDERBUFFER, rbo);
    glRenderbufferStorage(GL_RENDERBUFFER, GL_DEPTH_COMPONENT24, shadow_map_resolution, shadow_map_resolution);
    glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_RENDERBUFFER, rbo);

    if (glCheckFramebufferStatus(GL_DRAW_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE)
        throw std::runtime_error("Incomplete framebuffer!");

    glBindFramebuffer(GL_DRAW_FRAMEBUFFER, 0);

    const int point_shadow_texture_unit = 95;
    const unsigned int SHADOW_WIDTH = 1024, SHADOW_HEIGHT = 1024;
    GLuint depthMapFBO;
    glGenFramebuffers(1, &depthMapFBO);
    // create depth cubemap texture
    GLuint depthCubemap;
    glGenTextures(1, &depthCubemap);
    glBindTexture(GL_TEXTURE_CUBE_MAP, depthCubemap);
    for (unsigned int i = 0; i < 6; ++i)
        glTexImage2D(GL_TEXTURE_CUBE_MAP_POSITIVE_X + i,
                    0,
                    GL_DEPTH_COMPONENT,
                    SHADOW_WIDTH,
                    SHADOW_HEIGHT,
                    0,
                    GL_DEPTH_COMPONENT,
                    GL_FLOAT,
                    NULL);
    glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_WRAP_R, GL_CLAMP_TO_EDGE);
    // attach depth texture as FBO's depth buffer
    glBindFramebuffer(GL_FRAMEBUFFER, depthMapFBO);
    glFramebufferTexture(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, depthCubemap, 0);
    glDrawBuffer(GL_NONE);
    glReadBuffer(GL_NONE);
    if (glCheckFramebufferStatus(GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE)
        throw std::runtime_error("Incomplete framebuffer!");

    auto last_frame_start = std::chrono::high_resolution_clock::now();
    float time = 0.f;
    bool paused = false; //for what?
    std::map<SDL_Keycode, bool> button_down;

    glm::vec3 coords = {0.0, 0.0, 1.0};
    glm::vec3 camera_rotation = glm::vec3(0, -1.4, 0);
    glm::vec3 camera_direction, side_direction;
    const float cameraMovementSpeed = 1.f;
    const float cameraRotationSpeed = 1.f;

    bool running = true;
    while (running)
    {
        for (SDL_Event event; SDL_PollEvent(&event);) switch (event.type)
        {
        case SDL_QUIT:
            running = false;
            break;
        case SDL_WINDOWEVENT: switch (event.window.event)
            {
            case SDL_WINDOWEVENT_RESIZED:
                width = event.window.data1;
                height = event.window.data2;
                glViewport(0, 0, width, height);
                break;
            }
            break;
        case SDL_KEYDOWN:
            button_down[event.key.keysym.sym] = true;

            if (event.key.keysym.sym == SDLK_SPACE)
                paused = !paused;

            break;
        case SDL_KEYUP:
            button_down[event.key.keysym.sym] = false;
            break;
        }

        if (!running)
            break;


        auto now = std::chrono::high_resolution_clock::now();
        float dt = std::chrono::duration_cast<std::chrono::duration<float>>(now - last_frame_start).count();
        last_frame_start = now;

        if (!paused)
            time += dt;
        
        if (button_down[SDLK_LEFT])
            camera_rotation[1] -= cameraRotationSpeed * dt;
        if (button_down[SDLK_RIGHT])
            camera_rotation[1] += cameraRotationSpeed * dt;
        if (button_down[SDLK_UP])
            camera_rotation[0] -= cameraRotationSpeed * dt;
        if (button_down[SDLK_DOWN])
            camera_rotation[0] += cameraRotationSpeed * dt;

        camera_direction = glm::vec4(0, 0, 10, 1) * rotation_matrix(camera_rotation);

        if (button_down[SDLK_w])
            coords += camera_direction;
        if (button_down[SDLK_s])
            coords -= camera_direction;

        side_direction = glm::vec4(10, 0, 0, 1) * rotation_matrix(camera_rotation);

        if (button_down[SDLK_a])
            coords += side_direction;
        if (button_down[SDLK_d])
            coords -= side_direction;

        glm::vec3 light_direction = glm::normalize(glm::vec3(std::cos(time * 0.125f), 1.f, std::sin(time * 0.125f))); //maybe change time dependancy
        glm::vec3 light_z = -light_direction;
        glm::vec3 light_x = glm::normalize(glm::cross(light_z, {0.f, 1.f, 0.f}));
        glm::vec3 light_y = glm::cross(light_x, light_z);

        float dx = -std::numeric_limits<float>::infinity();
        float dy = -std::numeric_limits<float>::infinity();
        float dz = -std::numeric_limits<float>::infinity();
        for(auto _v : bounding_box) {
            glm::vec3 v = _v - c;
            dx = std::max(dx, glm::dot(v, light_x));
            dy = std::max(dy, glm::dot(v, light_y));
            dz = std::max(dz, glm::dot(v, light_z));
        }
        glm::mat4 transform = glm::inverse(glm::mat4({
            {dx * light_x.x, dx * light_x.y, dx * light_x.z, 0.f},
            {dy * light_y.x, dy * light_y.y, dy * light_y.z, 0.f},
            {dz * light_z.x, dz * light_z.y, dz * light_z.z, 0.f},
            {c.x, c.y, c.z, 1.f}
        }));

        glm::mat4 model(1.f);

        auto point_light_position = glm::vec3(900.0f, 150.0f, 350.0f);

        // *** Точечный источник, тень
        float near2 = 0.01;
        float far2 = 1000.f;
        glm::mat4
            shadowProj = glm::perspective(glm::radians(90.0f), (float) SHADOW_WIDTH / (float) SHADOW_HEIGHT, near2, far2);
        std::vector<glm::mat4> shadowTransforms;
        shadowTransforms.push_back(shadowProj * glm::lookAt(point_light_position,
                                                            point_light_position + glm::vec3(1.0f, 0.0f, 0.0f),
                                                            glm::vec3(0.0f, -1.0f, 0.0f)));
        shadowTransforms.push_back(shadowProj * glm::lookAt(point_light_position,
                                                            point_light_position + glm::vec3(-1.0f, 0.0f, 0.0f),
                                                            glm::vec3(0.0f, -1.0f, 0.0f)));
        shadowTransforms.push_back(shadowProj * glm::lookAt(point_light_position,
                                                            point_light_position + glm::vec3(0.0f, 1.0f, 0.0f),
                                                            glm::vec3(0.0f, 0.0f, 1.0f)));
        shadowTransforms.push_back(shadowProj * glm::lookAt(point_light_position,
                                                            point_light_position + glm::vec3(0.0f, -1.0f, 0.0f),
                                                            glm::vec3(0.0f, 0.0f, -1.0f)));
        shadowTransforms.push_back(shadowProj * glm::lookAt(point_light_position,
                                                            point_light_position + glm::vec3(0.0f, 0.0f, 1.0f),
                                                            glm::vec3(0.0f, -1.0f, 0.0f)));
        shadowTransforms.push_back(shadowProj * glm::lookAt(point_light_position,
                                                            point_light_position + glm::vec3(0.0f, 0.0f, -1.0f),
                                                            glm::vec3(0.0f, -1.0f, 0.0f)));

        glBindFramebuffer(GL_FRAMEBUFFER, depthMapFBO);
        glViewport(0, 0, SHADOW_WIDTH, SHADOW_HEIGHT);
        glClear(GL_DEPTH_BUFFER_BIT);
        glEnable(GL_DEPTH_TEST);
        glDepthFunc(GL_LEQUAL);
        glEnable(GL_CULL_FACE);
        glCullFace(GL_BACK);

        glClearColor(0.1f, 0.8f, 0.1f, 1.0f);
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

        glUseProgram(point_shadow_program);
        glUniformMatrix4fv(point_shadow_model_location, 1, GL_FALSE, reinterpret_cast<const GLfloat *>(&model));
        glUniform1f(point_shadow_far_plane_location, far2);
        glUniform3f(point_shadow_light_pos_location,
                    point_light_position.x,
                    point_light_position.y,
                    point_light_position.z);

        for (size_t i = 0; i < 6; ++i) {
            std::string name = "shadowMatrices[" + std::to_string(i) + "]";
            auto loc = glGetUniformLocation(point_shadow_program, name.data());
            glUniformMatrix4fv(loc, 1, GL_FALSE,
                                reinterpret_cast<GLfloat *>(&shadowTransforms[i]));
        }
        glUniformMatrix4fv(point_shadow_shadow_matrices_location, 6, GL_FALSE,
                        reinterpret_cast<const GLfloat *>(shadowTransforms.data()));
        glBindVertexArray(vao);

        int first = 0;
        for (size_t s = 0; s < shapes.size(); s++) {
            glDrawArrays(GL_TRIANGLES, first, (GLint)shapes[s].mesh.indices.size());
            first += shapes[s].mesh.indices.size();
        }

        glBindFramebuffer(GL_DRAW_FRAMEBUFFER, shadow_fbo);
        glClearColor(1.f, 1.f, 0.f, 0.f);
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
        glViewport(0, 0, shadow_map_resolution, shadow_map_resolution);
        glEnable(GL_DEPTH_TEST);
        glDepthFunc(GL_LEQUAL);
        glEnable(GL_CULL_FACE);
        glCullFace(GL_BACK);

        glUseProgram(shadow_program);
        glUniformMatrix4fv(shadow_model_location, 1, GL_FALSE, reinterpret_cast<float *>(&model));
        glUniformMatrix4fv(shadow_transform_location, 1, GL_FALSE, reinterpret_cast<float *>(&transform));

        glBindVertexArray(vao);

        first = 0;
        for (size_t s = 0; s < shapes.size(); s++) {
            int id = shapes[s].mesh.material_ids[0];
            std::string texture_path = materials_dir + materials[id].alpha_texname;

            std::replace(texture_path.begin(), texture_path.end(), '\\', '/');

            glUniform1i(shadow_alpha_texture_location, textures.get_texture(texture_path));
            glUniform1i(shadow_flag_map_d_location, !materials[id].alpha_texname.empty());

            glDrawArrays(GL_TRIANGLES, first, (GLint)shapes[s].mesh.indices.size());
            first += shapes[s].mesh.indices.size();
        }

        glBindFramebuffer(GL_DRAW_FRAMEBUFFER, 0);
        glViewport(0, 0, width, height);
        glClearColor(0.8f, 0.8f, 0.9f, 0.f);
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
        glEnable(GL_DEPTH_TEST);
        glDepthFunc(GL_LEQUAL);
        glEnable(GL_CULL_FACE);
        glCullFace(GL_BACK);
        
        float near = 0.1f;
        float far = 100000.f;

        glm::mat4 view(1.f);
        view = glm::rotate(view, camera_rotation.x, {1, 0, 0});
        view = glm::rotate(view, camera_rotation.y, {0, 1, 0});
        view = glm::rotate(view, camera_rotation.z, {0, 0, 1});
        view = glm::translate(view, coords);
        glm::mat4 projection = glm::perspective(glm::pi<float>() / 3.f, (width * 1.f) / height, near, far);
        glm::vec3 camera_position = (glm::inverse(view) * glm::vec4(0.f, 0.f, 0.f, 1.f)).xyz();

        glUseProgram(program);
        glUniformMatrix4fv(model_location, 1, GL_FALSE, reinterpret_cast<float *>(&model));
        glUniformMatrix4fv(view_location, 1, GL_FALSE, reinterpret_cast<float *>(&view));
        glUniformMatrix4fv(projection_location, 1, GL_FALSE, reinterpret_cast<float *>(&projection));
        glUniformMatrix4fv(transform_location, 1, GL_FALSE, reinterpret_cast<float *>(&transform));
        glUniform3fv(light_direction_location, 1, reinterpret_cast<float *>(&light_direction));
        glUniform3fv(camera_position_location, 1, reinterpret_cast<float *>(&camera_position));
        glUniform3f(light_color_location, 0.7f, 0.6f, 0.2f);
        glUniform3f(ambient_location, 0.5f, 0.5f, 0.5f);
        glUniform1i(shadow_map_location, 1);

        glActiveTexture(GL_TEXTURE0 + point_shadow_texture_unit);
        glBindTexture(GL_TEXTURE_CUBE_MAP, depthCubemap);

        glUniform3f(point_light_position_location, point_light_position.x, point_light_position.y, point_light_position.z);
        glUniform3f(point_light_color_location, 0.f, 0.f, 1.2f);
        glUniform3f(point_light_attenuation_location, 0.01f, 0.001f, 0.0001f);
        glUniform1i(depthMap_location, point_shadow_texture_unit);
        glUniform1f(far_plane_location, far2);

        first = 0;
        for (size_t s = 0; s < shapes.size(); s++) {
            auto pupupu = convertToSet(shapes[s].mesh.material_ids);
            if (pupupu.size() == 2) {
                int flag;
                for (int i = 1; i < shapes[s].mesh.material_ids.size(); i += 1) {
                    if (shapes[s].mesh.material_ids[i] != shapes[s].mesh.material_ids[i - 1]) {
                        flag = i;
                        break;
                    }
                }

                int id = shapes[s].mesh.material_ids[flag-1];
                std::string texture_path = materials_dir + materials[id].ambient_texname;
                std::replace(texture_path.begin(), texture_path.end(), '\\', '/');

                glUniform1i(ambient_texture_location, textures.get_texture(texture_path, true));
                glUniform1f(glossiness_location, materials[id].specular[0]);
                glUniform1f(power_location, materials[id].shininess);

                texture_path = materials_dir + materials[id].alpha_texname;
                std::replace(texture_path.begin(), texture_path.end(), '\\', '/');

                glUniform1i(alpha_texture_location, textures.get_texture(texture_path));
                glUniform1i(flag_map_d_location, !materials[id].alpha_texname.empty());

                texture_path = materials_dir + materials[id].bump_texname;
                std::replace(texture_path.begin(), texture_path.end(), '\\', '/');

                glUniform1i(bump_texture_location, textures.get_texture(texture_path));
                glUniform1i(flag_map_bump_location, !materials[id].bump_texname.empty());

                texture_path = materials_dir + materials[id].specular_texname;
                std::replace(texture_path.begin(), texture_path.end(), '\\', '/');

                glUniform1i(specular_texture_location, textures.get_texture(texture_path));
                glUniform1i(flag_map_Ks_location, !materials[id].specular_texname.empty());

                glDrawArrays(GL_TRIANGLES, first, (GLint)(flag*3));
                first += flag * 3;

                id = shapes[s].mesh.material_ids[flag];
                texture_path = materials_dir + materials[id].ambient_texname;
                std::replace(texture_path.begin(), texture_path.end(), '\\', '/');

                glUniform1i(ambient_texture_location, textures.get_texture(texture_path, true));
                glUniform1f(glossiness_location, materials[id].specular[0]);
                glUniform1f(power_location, materials[id].shininess);

                texture_path = materials_dir + materials[id].alpha_texname;
                std::replace(texture_path.begin(), texture_path.end(), '\\', '/');

                glUniform1i(alpha_texture_location, textures.get_texture(texture_path));
                glUniform1i(flag_map_d_location, !materials[id].alpha_texname.empty());

                texture_path = materials_dir + materials[id].bump_texname;
                std::replace(texture_path.begin(), texture_path.end(), '\\', '/');

                glUniform1i(bump_texture_location, textures.get_texture(texture_path));
                glUniform1i(flag_map_bump_location, !materials[id].bump_texname.empty());

                texture_path = materials_dir + materials[id].specular_texname;
                std::replace(texture_path.begin(), texture_path.end(), '\\', '/');

                glUniform1i(specular_texture_location, textures.get_texture(texture_path));
                glUniform1i(flag_map_Ks_location, !materials[id].specular_texname.empty());

                glDrawArrays(GL_TRIANGLES, first, (GLint)(shapes[s].mesh.indices.size() - flag * 3));
                first += (shapes[s].mesh.indices.size() - flag * 3);
            } else if (pupupu.size() == 1) {
                int id = shapes[s].mesh.material_ids[0];
                std::string texture_path = materials_dir + materials[id].ambient_texname;
                std::replace(texture_path.begin(), texture_path.end(), '\\', '/');

                glUniform1i(ambient_texture_location, textures.get_texture(texture_path, true));
                glUniform1f(glossiness_location, materials[id].specular[0]);
                glUniform1f(power_location, materials[id].shininess);

                texture_path = materials_dir + materials[id].alpha_texname;
                std::replace(texture_path.begin(), texture_path.end(), '\\', '/');

                glUniform1i(alpha_texture_location, textures.get_texture(texture_path));
                glUniform1i(flag_map_d_location, !materials[id].alpha_texname.empty());

                texture_path = materials_dir + materials[id].bump_texname;
                std::replace(texture_path.begin(), texture_path.end(), '\\', '/');

                glUniform1i(bump_texture_location, textures.get_texture(texture_path));
                glUniform1i(flag_map_bump_location, !materials[id].bump_texname.empty());

                texture_path = materials_dir + materials[id].specular_texname;
                std::replace(texture_path.begin(), texture_path.end(), '\\', '/');

                glUniform1i(specular_texture_location, textures.get_texture(texture_path));
                glUniform1i(flag_map_Ks_location, !materials[id].specular_texname.empty());

                glDrawArrays(GL_TRIANGLES, first, (GLint)shapes[s].mesh.indices.size());
                first += shapes[s].mesh.indices.size();
            } else {
                std::cout << "pupupu.size() = " << pupupu.size() << std::endl;
                return EXIT_FAILURE;
            }
        }
        glBindFramebuffer(GL_FRAMEBUFFER, 0);
        SDL_GL_SwapWindow(window);
    }    
    SDL_GL_DeleteContext(gl_context);
    SDL_DestroyWindow(window);    
}
catch (std::exception const & e)
{
    std::cerr << e.what() << std::endl;
    return EXIT_FAILURE;
}