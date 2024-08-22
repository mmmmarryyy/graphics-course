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
    texcoord = vec2(in_texcoord.x, 1.f-in_texcoord.y);
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
    texcoord = vec2(in_texcoord.x, 1.0 - in_texcoord.y);
}
)";

const char fragment_shader_source[] = //don't forget to add point_color (example in practice7)
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

uniform vec3 point_light_position;
uniform vec3 point_light_color;
uniform vec3 point_light_attenuation;

uniform samplerCube depthMap;
uniform float far_plane;

in vec3 position;
in vec3 normal;
in vec2 texcoord;

layout (location = 0) out vec4 out_color;

float diffuse(vec3 direction) {
    return max(0.0, dot(normal, direction));
}

float specular(vec3 direction) {
    vec3 reflected_direction = 2.0 * normal * dot(normal, direction) - direction;
    vec3 view_direction = normalize(camera_position - position);
    return glossiness * pow(max(0.0, dot(reflected_direction, view_direction)), power);
}

float phong(vec3 direction) {
    return diffuse(direction) + specular(direction);
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

void main()
{
    if (flag_map_d == 1) {
        if (texture(alpha_texture, texcoord).r < 0.5) {
            discard;
        }
    }

    vec4 shadow_pos = transform * vec4(position, 1.0);
    shadow_pos /= shadow_pos.w;
    shadow_pos = shadow_pos * 0.5 + vec4(0.5);
    
    bool in_shadow_texture = (shadow_pos.x >= 0.0) && (shadow_pos.x < 1.0) && 
        (shadow_pos.y >= 0.0) && (shadow_pos.y < 1.0) && 
        (shadow_pos.z >= 0.0) && (shadow_pos.z < 1.0);

    float factor = 1.0;

    // if (in_shadow_texture) {
    // vec2 sum = vec2(0.0);
    // float sum_w = 0.0;
    // const int N = 2;
    // float radius = 3.0;
    // for (int x = -N; x <= N; ++x) {
    //     for (int y = -N; y <= N; ++y) {
    //         float c = exp(-float(x*x + y*y) / (radius*radius));
    //         sum += c * texture(shadow_map, shadow_pos.xy + vec2(x,y) / vec2(textureSize(shadow_map, 0))).xy;
    //         sum_w += c;
    //     }
    // }


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
    vec3 point_color = phong(point_light_direction) * point_light_color / (point_light_attenuation.x + point_light_attenuation.y * point_light_distance + point_light_attenuation.z * point_light_distance * point_light_distance);
    float point_shadow = ShadowCalculation();

    vec3 light = albedo * ambient;
    light += light_color * phong(light_direction) * shadow_factor; //here need to add point color
    light += point_color * point_shadow;
    vec3 color = light;
    out_color = vec4(color, 1.0);
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

// GLuint create_program(GLuint vertex_shader, GLuint fragment_shader)
// {
//     GLuint result = glCreateProgram();
//     glAttachShader(result, vertex_shader);
//     glAttachShader(result, fragment_shader);
//     glLinkProgram(result);

//     GLint status;
//     glGetProgramiv(result, GL_LINK_STATUS, &status);
//     if (status != GL_TRUE)
//     {
//         GLint info_log_length;
//         glGetProgramiv(result, GL_INFO_LOG_LENGTH, &info_log_length);
//         std::string info_log(info_log_length, '\0');
//         glGetProgramInfoLog(result, info_log.size(), nullptr, info_log.data());
//         throw std::runtime_error("Program linkage failed: " + info_log);
//     }

//     return result;
// }

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

// ------------------------------------------------------------------------------------------------------------

// ------------------------------------------------------------------------------------------------------------
    

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
    GLint load_texture(const std::string &path);
    GLint get_texture(const std::string &path);

private:
    GLint m_first_unit;
    std::unordered_map<std::string, std::pair<GLuint, GLint>> m_textures;
};

GLint texture_holder::load_texture(const std::string &path) {
    GLint unit = m_first_unit + (GLint)m_textures.size();
    // if (unit > 15) {
    //     unit = 15;
    // }
    auto it = m_textures.find(path);
    if(it != m_textures.end()) return it->second.second;
    glGenTextures(1, &m_textures[path].first);
    m_textures[path].second = unit;
    glActiveTexture(GL_TEXTURE0 + unit);
    std::cout << "unit1 = " << unit << std::endl;
    glBindTexture(GL_TEXTURE_2D, m_textures[path].first);
    glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR);
    glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    int x, y, channels_in_file;
    auto pixels = stbi_load(path.c_str(), &x, &y, &channels_in_file, 4);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, x, y, 0, GL_RGBA, GL_UNSIGNED_BYTE, pixels);
    glGenerateMipmap(GL_TEXTURE_2D);
    stbi_image_free(pixels);
    std::cout << "unit2 = " << unit << std::endl;
    std::cout << "path = " << path << std::endl << std::endl;
    return unit;
}

GLint texture_holder::get_texture(const std::string &path) {
    // std::cout << "get_texture, path = " << path << std::endl;
    auto it = m_textures.find(path);
    if (it == m_textures.end()) {
        // std::cout << "don't find texture" << std::endl;
        return load_texture(path);
    }
    // if (it->second.second > 26) {
    //     // std::cout << "path = " << path << std::endl;
    //     // std::cout << "it->second.second = " << it->second.second << std::endl;
    // }
    auto unit = it->second.second;
    glActiveTexture(GL_TEXTURE0 + unit);
    return unit;
    // else if (it->second.second < 15) {
    //     return it->second.second;
    // } else {
    //     // std::cout << "hello" << std::endl;
    //     // return it->second.second;
    //     glGenTextures(1, &m_textures[path].first);
    //     m_textures[path].second = 15;
    //     int unit = 15;
    //     glActiveTexture(GL_TEXTURE0 + unit);
    //     std::cout << "unit = " << unit << std::endl;
    //     glBindTexture(GL_TEXTURE_2D, m_textures[path].first);
    //     glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR);
    //     glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    //     int x, y, channels_in_file;
    //     auto pixels = stbi_load(path.c_str(), &x, &y, &channels_in_file, 4);
    //     glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, x, y, 0, GL_RGBA, GL_UNSIGNED_BYTE, pixels);
    //     glGenerateMipmap(GL_TEXTURE_2D);
    //     stbi_image_free(pixels);
    //     return 0;
    // };
}

texture_holder::texture_holder(GLint first_unit) : m_first_unit(first_unit) {}

// ------------------------------------------------------------------------------------------------------------

std::set<int> convertToSet(std::vector<int> v)
{
    // Declaring the set
    std::set<int> s;
 
    // Traverse the Vector
    for (int x : v) {
 
        // Insert each element
        // into the Set
        s.insert(x);
    }
 
    // Return the resultant Set
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

    SDL_Window * window = SDL_CreateWindow("Homework 2",
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

    // ------------------------------------------------------------------------------------------------------------
    
    // ------------------------------------------------------------------------------------------------------------
    obj_data scene;
    // Loop over shapes
    for (size_t s = 0; s < shapes.size(); s++) {
        // // Loop over faces(polygon)
        // size_t index_offset = 0;
        // for (size_t f = 0; f < shapes[s].mesh.num_face_vertices.size(); f++) {
        //     size_t fv = size_t(shapes[s].mesh.num_face_vertices[f]);


        //     // Loop over vertices in the face.
        //     for (size_t v = 0; v < fv; v++) {
        //         scene.vertices.push_back(obj_data::vertex());

        //         // access to vertex
        //         tinyobj::index_t idx = shapes[s].mesh.indices[index_offset + v];

        //         tinyobj::real_t vx = attrib.vertices[3*size_t(idx.vertex_index)+0];
        //         tinyobj::real_t vy = attrib.vertices[3*size_t(idx.vertex_index)+1];
        //         tinyobj::real_t vz = attrib.vertices[3*size_t(idx.vertex_index)+2];
        //         scene.vertices.back().position = {vx, vy, vz};

        //         // Check if `normal_index` is zero or positive. negative = no normal data
        //         if (idx.normal_index >= 0) {
        //             tinyobj::real_t nx = attrib.normals[3*size_t(idx.normal_index)+0];
        //             tinyobj::real_t ny = attrib.normals[3*size_t(idx.normal_index)+1];
        //             tinyobj::real_t nz = attrib.normals[3*size_t(idx.normal_index)+2];

        //             scene.vertices.back().normal = {nx, ny, nz};
        //         }

        //         // Check if `texcoord_index` is zero or positive. negative = no texcoord data
        //         if (idx.texcoord_index >= 0) {
        //             tinyobj::real_t tx = attrib.texcoords[2*size_t(idx.texcoord_index)+0];
        //             tinyobj::real_t ty = attrib.texcoords[2*size_t(idx.texcoord_index)+1];

        //             scene.vertices.back().texcoord = {tx, ty};
        //         }
        //         // Optional: vertex colors
        //         // tinyobj::real_t red   = attrib.colors[3*size_t(idx.vertex_index)+0];
        //         // tinyobj::real_t green = attrib.colors[3*size_t(idx.vertex_index)+1];
        //         // tinyobj::real_t blue  = attrib.colors[3*size_t(idx.vertex_index)+2];
        //     }
        //     index_offset += fv;

        //     // per-face material
        //     shapes[s].mesh.material_ids[f];
        // }

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

    // std::reverse(materials.begin(), materials.end());
    // std::shuffle(materials.begin(), materials.end(), g);
    // std::cout << "materials.size() = " << materials.size() << std::endl;
    // for(auto &material : materials) {
    //     if (material.alpha_texname.empty() == 0) {
    //         std::cout << std::endl << "INSIDE material.alpha_texname" << std::endl;
    //         std::string texture_path = materials_dir + material.alpha_texname;
    //         std::replace(texture_path.begin(), texture_path.end(), '\\', '/');

    //         std::cout << "texture_path = " << texture_path << std::endl;
    //         std::cout << "counter = " << counter++ << std::endl << std::endl << std::endl;
    //         textures.load_texture(texture_path);
    //     }
    // }
    for(auto &material : materials) {
        std::string texture_path = materials_dir + material.ambient_texname;
        std::replace(texture_path.begin(), texture_path.end(), '\\', '/');

        // std::cout << "texture_path = " << texture_path << std::endl;
        // std::cout << "counter = " << counter++ << std::endl << std::endl;
        textures.load_texture(texture_path);

        // if (material.alpha_texname.empty() == 0) {
        //     std::cout << std::endl << "INSIDE material.alpha_texname" << std::endl;
        //     texture_path = materials_dir + material.alpha_texname;
        //     std::replace(texture_path.begin(), texture_path.end(), '\\', '/');

        //     std::cout << "texture_path = " << texture_path << std::endl;
        //     std::cout << "counter = " << counter++ << std::endl << std::endl << std::endl;
        //     textures.load_texture(texture_path);
        // }
    }
    int MaxTextureImageUnits;
    glGetIntegerv(GL_MAX_TEXTURE_IMAGE_UNITS, &MaxTextureImageUnits);
    std::cout << "MaxTextureImageUnits = " << MaxTextureImageUnits << std::endl;
    auto bounding_box = get_bounding_box(scene.vertices);
    glm::vec3 c = std::accumulate(bounding_box.begin(), bounding_box.end(), glm::vec3(0.f)) / 8.f;
    // ------------------------------------------------------------------------------------------------------------

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

    // ------------------------------------------------------------------------------------------------------------
    
    // ------------------------------------------------------------------------------------------------------------
    
    
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

    // ------------------------------------------------------------------------------------------------------------

    // ------------------------------------------------------------------------------------------------------------
    
    // ------------------------------------------------------------------------------------------------------------
    
    // int last_textrue_location_id = 1; // 0/1 will be free
    // std::map<std::string, int> texture_pos;

    // glUseProgram(program);
    // for (size_t s = 0; s < shapes.size(); s++) {
    //     int id = shapes[s].mesh.material_ids[0];
    //     std::string ambient_texname = materials[id].ambient_texname;

    //     if(texture_pos.find(ambient_texname)!=texture_pos.end()) continue;
    //     texture_pos[ambient_texname] = ++last_textrue_location_id;

    //     std::cout << "ambient_texname = " << ambient_texname << std::endl;

    //     GLuint textureID=0;
    //     glGenTextures(1, &textureID);
    //     glActiveTexture(GL_TEXTURE0 + last_textrue_location_id);
    //     glBindTexture(GL_TEXTURE_2D, textureID);
    //     glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR);
    //     glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);

    //     int texture_width, texture_height, texture_nrChannels;
        
    //     for(char &c : ambient_texname) if(c=='\\') c='/';
    //     std::string path = project_root + "/" + ambient_texname;
    //     unsigned char* pixels_texture = stbi_load(path.c_str(), &texture_width, &texture_height, &texture_nrChannels, 4);
                    
    //     glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, texture_width, texture_height, 0, GL_RGBA, GL_UNSIGNED_BYTE, pixels_texture);

    //     glGenerateMipmap(GL_TEXTURE_2D);

    //     stbi_image_free(pixels_texture);        
    // }

    // std::cout << std::endl;


    // std::map<std::string, int> alpha_texture_pos;
    // for (size_t s = 0; s < shapes.size(); s++) {
    //     int id = shapes[s].mesh.material_ids[0];
    //     std::string alpha_texname = materials[id].alpha_texname;

    //     if(alpha_texture_pos.find(alpha_texname)!=alpha_texture_pos.end()) continue;
    //     alpha_texture_pos[alpha_texname] = ++last_textrue_location_id;

    //     std::cout << "alpha_texname = " << alpha_texname << std::endl;

    //     GLuint textureID=0;
    //     glGenTextures(1, &textureID);
    //     glActiveTexture(GL_TEXTURE0 + last_textrue_location_id);
    //     glBindTexture(GL_TEXTURE_2D, textureID);
    //     glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR);
    //     glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);

    //     int texture_width, texture_height, texture_nrChannels;
        
    //     for(char &c : alpha_texname) if(c=='\\') c='/';
    //     std::string path = project_root + "/" + alpha_texname;
    //     unsigned char* pixels_texture = stbi_load(path.c_str(), &texture_width, &texture_height, &texture_nrChannels, 4);
                    
    //     glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, texture_width, texture_height, 0, GL_RGBA, GL_UNSIGNED_BYTE, pixels_texture);

    //     glGenerateMipmap(GL_TEXTURE_2D);

    //     stbi_image_free(pixels_texture);        
    // }

// ------------------------------------------------------------------------------------------------------------


 // ------------------------------------------------------------------------------------------------------------
    
    // ------------------------------------------------------------------------------------------------------------
    
    //seems like it is bounding box

    // float infty = std::numeric_limits<float>::infinity();
    // float min_x = infty, max_x = -infty;
    // float min_y = infty, max_y = -infty;
    // float min_z = infty, max_z = -infty;
    // for(obj_data::vertex el : scene.vertices) {
    //     min_x = std::min(min_x, el.position[0]);
    //     max_x = std::max(max_x, el.position[0]);

    //     min_y = std::min(min_y, el.position[1]);
    //     max_y = std::max(max_y, el.position[1]);

    //     min_z = std::min(min_z, el.position[2]);
    //     max_z = std::max(max_z, el.position[2]);
    // }

    // std::vector<std::vector<float>> v(8, std::vector<float>(3));
    // v = {
    //     {min_x, min_y, min_z},
    //     {min_x, min_y, max_z},
    //     {min_x, max_y, min_z},
    //     {min_x, max_y, max_z},
    //     {max_x, min_y, min_z},
    //     {max_x, min_y, max_z},
    //     {max_x, max_y, min_z},
    //     {max_x, max_y, max_z},
    // };


// ------------------------------------------------------------------------------------------------------------

    auto last_frame_start = std::chrono::high_resolution_clock::now();
    float time = 0.f;
    bool paused = false; //for what?
    std::map<SDL_Keycode, bool> button_down;

    // ------------------------------------------------------------------------------------------------------------
    
    // ------------------------------------------------------------------------------------------------------------
    
    // glm::vec3 cameraPos   = glm::vec3(0.0f, 0.0f,  3.0f);
    // glm::vec3 cameraFront = glm::vec3(0.0f, 0.0f, -1.0f);
    // glm::vec3 cameraUp    = glm::vec3(0.0f, 1.0f,  0.0f);
    // glm::vec3 direction;
    // float yaw = -90.f, pitch = 0.f;
    glm::vec3 coords = {0.0, 0.0, 1.0};
    glm::vec3 camera_rotation = glm::vec3(0, -1.4, 0);
    glm::vec3 camera_direction, side_direction;
    const float cameraMovementSpeed = 1.f;
    const float cameraRotationSpeed = 1.f;

    // float view_angle = glm::pi<float>() / 6.f;
    // float camera_distance = 3.5f;
    // float camera_rotation = glm::pi<float>() / 6.f;

    // ------------------------------------------------------------------------------------------------------------

    // for (size_t s = 0; s < shapes.size(); s++) {
            // int id = shapes[s].mesh.material_ids[0];
            // std::string texture_path = materials_dir + materials[id].ambient_texname;
            // std::replace(texture_path.begin(), texture_path.end(), '\\', '/');

            // if (texture_path == "/Users/maria.barkovskaya/Documents/uni/comp_graph/graphics-course-practice/hw2/textures/vase_plant.png") {
            //     std::cout << "s = " << s << std::endl;
            //     std::cout << "material_ids.size() = " << shapes[s].mesh.material_ids.size() << std::endl;
            //     std::cout << "texture_path = " << texture_path << std::endl;
            // } else {
            //     // std::cout << "texture_path = " << texture_path << std::endl;
            // }
            // auto pupupu = convertToSet(shapes[s].mesh.material_ids);
        //     if (pupupu.size() > 1) {
        //             std::cout << shapes[s].name << std::endl;
        //             std::cout << "s = " << s << std::endl;
        //             for (auto i: pupupu) {
        //                 std::cout << "i = " << i << std::endl;
        //                 std::cout << "materials[id].ambient_texname = " << materials[i].ambient_texname << std::endl;
        //             }

        //             for (int i = 3; i < shapes[s].mesh.material_ids.size(); i+=3) {
        //                 if (shapes[s].mesh.material_ids[i] == shapes[s].mesh.material_ids[i - 1]) {
        //                     // continue;
        //                 } else {
        //                     std::cout << i << std::endl;
        //                     std::cout << "materials[id].ambient_texname = " << materials[shapes[s].mesh.material_ids[i]].ambient_texname << std::endl;
        //                     std::cout << "materials[id].ambient_texname = " << materials[shapes[s].mesh.material_ids[i-1]].ambient_texname << std::endl;
        //                     std::cout << std::endl;
        //                     // std::cout << "num_face_vertices[id] = " << shapes[1].mesh.num_face_vertices[i] << std::endl;
        //                 }
        //             }

        //             std::cout << std::endl;
        //     }
        // }

    // std::cout << shapes[1].name << std::endl;
    // auto pupupu = convertToSet(shapes[1].mesh.material_ids);
    // for (auto i: pupupu) {
    //     std::cout << "i = " << i << std::endl;
    // }

    // std::cout << shapes[1].mesh.material_ids.size() << std::endl;
    // std::cout << shapes[1].mesh.indices.size() << std::endl;
    // std::cout << shapes[1].mesh.num_face_vertices.size() << std::endl;
    // std::cout << shapes[1].mesh.tags.size() << std::endl;

    // for (int i = 3; i < shapes[1].mesh.material_ids.size(); i+=3) {
    //     if (shapes[1].mesh.material_ids[i] == shapes[1].mesh.material_ids[i - 1]) {
    //         // continue;
    //     } else {
    //         std::cout << i << std::endl;
    //         std::cout << "materials[id].ambient_texname = " << materials[shapes[1].mesh.material_ids[i]].ambient_texname << std::endl;
    //         std::cout << "materials[id].ambient_texname = " << materials[shapes[1].mesh.material_ids[i-1]].ambient_texname << std::endl;
    //         std::cout << std::endl;
    //         // std::cout << "num_face_vertices[id] = " << shapes[1].mesh.num_face_vertices[i] << std::endl;
    //     }
    // }

    // std::set<int> ind;

    // for (int i = 0; i < shapes[1].mesh.indices.size(); i++) {
    //     std::cout << shapes[1].mesh.indices[i].vertex_index << " ";
    //     ind.insert(i);
    // }

    // std::cout << ind.size() << std::endl;

    // return 0;

    // ------------------------------------------------------------------------------------------------------------

    std::cout << "before while (running)" << std::endl;
    std::cout << "MAX_TEXTURE_UNITS = " << GL_MAX_TEXTURE_UNITS << std::endl;
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

        int i = 0;
        //std::cout << "i = " << i++ << "; width = " << width << "; height = " << height << std::endl;
        // ------------------------------------------------------------------------------------------------------------
        
        // ------------------------------------------------------------------------------------------------------------
    
        if (button_down[SDLK_LEFT])
            camera_rotation[1] -= cameraRotationSpeed * dt;
        if (button_down[SDLK_RIGHT])
            camera_rotation[1] += cameraRotationSpeed * dt;
        if (button_down[SDLK_UP])
            camera_rotation[0] -= cameraRotationSpeed * dt;
        if (button_down[SDLK_DOWN])
            camera_rotation[0] += cameraRotationSpeed * dt;

        // if (button_down[SDLK_UP])
        //     view_angle -= 2.f * dt;
        // if (button_down[SDLK_DOWN])
        //     view_angle += 2.f * dt;

        // if (button_down[SDLK_LEFT])
        //     camera_rotation -= 2.f * dt;
        // if (button_down[SDLK_RIGHT])
        //     camera_rotation += 2.f * dt;

        // if (button_down[SDLK_w])
        //     camera_distance -= 100.f * dt;
        // if (button_down[SDLK_s])
        //     camera_distance += 100.f * dt;

        // if (pitch > 89.0f)
        //     pitch =  89.0f;
        // if (pitch < -89.0f)
        //     pitch = -89.0f;

        // direction.x = cos(glm::radians(yaw)) * cos(glm::radians(pitch));
        // direction.y = sin(glm::radians(pitch));
        // direction.z = sin(glm::radians(yaw)) * cos(glm::radians(pitch));
        // cameraFront = glm::normalize(direction);

        // // std::cout << "cameraPos1 = (" << cameraPos.x << ", " << cameraPos.y << ", " << cameraPos.z << ")" << std::endl;
        // if (button_down[SDLK_a]) {
        //     // std::cout << "inside if1" << std::endl;
        //     cameraPos -= glm::normalize(glm::cross(cameraFront, cameraUp)) * cameraMovementSpeed * dt;
        // } if (button_down[SDLK_d]) {
        //     // std::cout << "inside if2" << std::endl;
        //     cameraPos += glm::normalize(glm::cross(cameraFront, cameraUp)) * cameraMovementSpeed * dt;
        // } if (button_down[SDLK_w]) {
        //     // std::cout << "inside if3" << std::endl;
        //     cameraPos += cameraMovementSpeed * cameraFront * dt;
        // } if (button_down[SDLK_s]) {
        //     // std::cout << "inside if4" << std::endl;
        //     cameraPos -= cameraMovementSpeed * cameraFront * dt;
        // }

        // std::cout << "cameraPos2 = (" << cameraPos.x << ", " << cameraPos.y << ", " << cameraPos.z << ")" << std::endl;

        // ------------------------------------------------------------------------------------------------------------

        camera_direction = glm::vec4(0, 0, 10, 1) * rotation_matrix(camera_rotation);

        if (button_down[SDLK_w])
            coords += camera_direction;
        if (button_down[SDLK_s])
            coords -= camera_direction;

        //std::cout << "i = " << i++ << "; width = " << width << "; height = " << height << std::endl;

        side_direction = glm::vec4(10, 0, 0, 1) * rotation_matrix(camera_rotation);

        if (button_down[SDLK_a])
            coords += side_direction;
        if (button_down[SDLK_d])
            coords -= side_direction;

        //std::cout << "i = " << i++ << "; width = " << width << "; height = " << height << std::endl;

        // std::cout << "position = " << camera_position.x << " " << camera_position.y << " " << camera_position.z << std::endl;

        glm::vec3 light_direction = glm::normalize(glm::vec3(std::cos(time * 0.125f), 1.f, std::sin(time * 0.125f))); //maybe change time dependancy
        glm::vec3 light_z = -light_direction;
        glm::vec3 light_x = glm::normalize(glm::cross(light_z, {0.f, 1.f, 0.f}));
        glm::vec3 light_y = glm::cross(light_x, light_z);

        //std::cout << "i = " << i++ << "; width = " << width << "; height = " << height << std::endl;
        
        // ------------------------------------------------------------------------------------------------------------
        
        // ------------------------------------------------------------------------------------------------------------
    
        // float c_x = (max_x + min_x) / 2;
        // float c_y = (max_y + min_y) / 2;
        // float c_z = (max_z + min_z) / 2;
        // float light_x_mx = -infty;
        // for(auto &el : v) {
        //     light_x_mx = std::max(light_x_mx, std::abs(glm::dot({el[0]-c_x,el[1]-c_y,el[2]-c_z}, light_x)));
        // }
        // float light_y_mx = -infty;
        // for(auto &el : v) {
        //     light_y_mx = std::max(light_y_mx, std::abs(glm::dot({el[0]-c_x,el[1]-c_y,el[2]-c_z}, light_y)));
        // }
        // float light_z_mx = -infty;
        // for(auto &el : v) {
        //     light_z_mx = std::max(light_z_mx, std::abs(glm::dot({el[0]-c_x,el[1]-c_y,el[2]-c_z}, light_z)));
        // }
        // light_x *= light_x_mx;
        // light_y *= light_y_mx;
        // light_z *= light_z_mx;
        // glm::mat4 transform = glm::mat4(1.f);
        // transform = {
        //     {light_x[0], light_y[0], light_z[0], c_x},
        //     {light_x[1], light_y[1], light_z[1], c_y},
        //     {light_x[2], light_y[2], light_z[2], c_z},
        //     {0.0, 0.0, 0.0, 1.0}
        // };
        // transform = glm::inverse(glm::transpose(transform));

        // ------------------------------------------------------------------------------------------------------------

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

        //std::cout << "i = " << i++ << "; width = " << width << "; height = " << height << std::endl;

        // ------------------------------------------------------------------------------------------------------------
        
        // ------------------------------------------------------------------------------------------------------------
    
        
        

        auto point_light_position = glm::vec3(std::sin(time) * 200, 100.0f, std::cos(time) * 100);

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

        //std::cout << "i = " << i++ << "; width = " << width << "; height = " << height << std::endl;

        glBindFramebuffer(GL_FRAMEBUFFER, depthMapFBO);
        glViewport(0, 0, SHADOW_WIDTH, SHADOW_HEIGHT);
        glClear(GL_DEPTH_BUFFER_BIT);
        glEnable(GL_DEPTH_TEST);
        glDepthFunc(GL_LEQUAL);
        glEnable(GL_CULL_FACE);
        glCullFace(GL_BACK);

        //std::cout << "i = " << i++ << "; width = " << width << "; height = " << height << std::endl;

        glClearColor(0.1f, 0.8f, 0.1f, 1.0f);
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

        //std::cout << "i = " << i++ << "; width = " << width << "; height = " << height << std::endl;

        glUseProgram(point_shadow_program);
        glUniformMatrix4fv(point_shadow_model_location, 1, GL_FALSE, reinterpret_cast<const GLfloat *>(&model));
        glUniform1f(point_shadow_far_plane_location, far2);
        glUniform3f(point_shadow_light_pos_location,
                    point_light_position.x,
                    point_light_position.y,
                    point_light_position.z);

        //std::cout << "i = " << i++ << "; width = " << width << "; height = " << height << std::endl;
        for (size_t i = 0; i < 6; ++i) {
            std::string name = "shadowMatrices[" + std::to_string(i) + "]";
            auto loc = glGetUniformLocation(point_shadow_program, name.data());
            glUniformMatrix4fv(loc, 1, GL_FALSE,
                                reinterpret_cast<GLfloat *>(&shadowTransforms[i]));
        }
        glUniformMatrix4fv(point_shadow_shadow_matrices_location, 6, GL_FALSE,
                        reinterpret_cast<const GLfloat *>(shadowTransforms.data()));
        glBindVertexArray(vao);
        glDrawArrays(GL_TRIANGLES, 0, scene.vertices.size());

        //std::cout << "i = " << i++ << "; width = " << width << "; height = " << height << std::endl;

        // ------------------------------------------------------------------------------------------------------------

        // ------------------------------------------------------------------------------------------------------------
        
        // ------------------------------------------------------------------------------------------------------------
    
        
        glBindFramebuffer(GL_DRAW_FRAMEBUFFER, shadow_fbo);
        glClearColor(1.f, 1.f, 0.f, 0.f);
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
        //std::cout << "i = " << i++ << "; width = " << width << "; height = " << height << std::endl;
        glViewport(0, 0, shadow_map_resolution, shadow_map_resolution);
        //std::cout << "i = " << i++ << "; width = " << width << "; height = " << height << std::endl;
        glEnable(GL_DEPTH_TEST);
        glDepthFunc(GL_LEQUAL);
        glEnable(GL_CULL_FACE);
        glCullFace(GL_BACK);

        //std::cout << "i = " << i++ << "; width = " << width << "; height = " << height << std::endl;

        // ------------------------------------------------------------------------------------------------------------

        glUseProgram(shadow_program);
        glUniformMatrix4fv(shadow_model_location, 1, GL_FALSE, reinterpret_cast<float *>(&model));
        glUniformMatrix4fv(shadow_transform_location, 1, GL_FALSE, reinterpret_cast<float *>(&transform));

        glBindVertexArray(vao);

        //std::cout << "i = " << i++ << "; width = " << width << "; height = " << height << std::endl;
        
        int first = 0;
        for (size_t s = 0; s < shapes.size(); s++) {
            int id = shapes[s].mesh.material_ids[0];
            std::string texture_path = materials_dir + materials[id].alpha_texname;

            std::replace(texture_path.begin(), texture_path.end(), '\\', '/');

            glUniform1i(shadow_alpha_texture_location, textures.get_texture(texture_path));
            glUniform1i(shadow_flag_map_d_location, !materials[id].alpha_texname.empty());

            glDrawArrays(GL_TRIANGLES, first, (GLint)shapes[s].mesh.indices.size());
            first += shapes[s].mesh.indices.size();
        }

        //std::cout << "i = " << i++ << "; width = " << width << "; height = " << height << std::endl;

        // ------------------------------------------------------------------------------------------------------------
        
        // ------------------------------------------------------------------------------------------------------------

        // glBindTexture(GL_TEXTURE_2D, shadow_map);
        // glGenerateMipmap(GL_TEXTURE_2D);

        glBindFramebuffer(GL_DRAW_FRAMEBUFFER, 0);
        glViewport(0, 0, width, height);
        //std::cout << "i = " << i++ << "; width = " << width << "; height = " << height << std::endl;
        glClearColor(0.8f, 0.8f, 0.9f, 0.f); //why this?
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
        glEnable(GL_DEPTH_TEST);
        glDepthFunc(GL_LEQUAL);
        glEnable(GL_CULL_FACE);
        glCullFace(GL_BACK);

        //std::cout << "i = " << i++ << "; width = " << width << "; height = " << height << std::endl;
        
        // ------------------------------------------------------------------------------------------------------------

        float near = 0.1f;
        float far = 100000.f;

        //std::cout << "i = " << i++ << "; width = " << width << "; height = " << height << std::endl;

        glm::mat4 view(1.f);
        view = glm::rotate(view, camera_rotation.x, {1, 0, 0});
        view = glm::rotate(view, camera_rotation.y, {0, 1, 0});
        view = glm::rotate(view, camera_rotation.z, {0, 0, 1});
        view = glm::translate(view, coords);
        glm::mat4 projection = glm::perspective(glm::pi<float>() / 3.f, (width * 1.f) / height, near, far);
        glm::vec3 camera_position = (glm::inverse(view) * glm::vec4(0.f, 0.f, 0.f, 1.f)).xyz();

        // glm::mat4 view(1.f);
        // view = glm::translate(view, {0.f, 0.f, -camera_distance});
        // view = glm::rotate(view, view_angle, {1.f, 0.f, 0.f});
        // view = glm::rotate(view, camera_rotation, {0.f, 1.f, 0.f});
        // glm::mat4 projection = glm::perspective(glm::pi<float>() / 3.f, (width * 1.f) / height, near, far);
        // glm::vec3 camera_position = (glm::inverse(view) * glm::vec4(0.f, 0.f, 0.f, 1.f)).xyz();

        //std::cout << "i = " << i++ << "; width = " << width << "; height = " << height << std::endl;
        
        // glm::mat4 view(1.f);
        // view = glm::rotate(view, camera_rotation.x, {1, 0, 0});
        // view = glm::rotate(view, camera_rotation.y, {0, 1, 0});
        // view = glm::rotate(view, camera_rotation.z, {0, 0, 1});
        // view = glm::translate(view, camera_position);
        // glm::mat4 projection = glm::mat4(1.f);
        // projection = glm::perspective(glm::pi<float>() / 2.f, (1.f * width) / height, near, far);
        // glm::vec3 camera_position = (glm::inverse(view) * glm::vec4(0.f, 0.f, 0.f, 1.f)).xyz();
        

        glUseProgram(program);
        glUniformMatrix4fv(model_location, 1, GL_FALSE, reinterpret_cast<float *>(&model));
        glUniformMatrix4fv(view_location, 1, GL_FALSE, reinterpret_cast<float *>(&view));
        glUniformMatrix4fv(projection_location, 1, GL_FALSE, reinterpret_cast<float *>(&projection));
        glUniformMatrix4fv(transform_location, 1, GL_FALSE, reinterpret_cast<float *>(&transform));

        //std::cout << "i = " << i++ << "; width = " << width << "; height = " << height << std::endl;
        glUniform3fv(light_direction_location, 1, reinterpret_cast<float *>(&light_direction));
        glUniform3fv(camera_position_location, 1, reinterpret_cast<float *>(&camera_position));
        glUniform3f(light_color_location, 0.8f, 0.7f, 0.6f);
        glUniform3f(ambient_location, 0.3f, 0.3f, 0.3f);
        glUniform1i(shadow_map_location, 1);
        //std::cout << "i = " << i++ << "; width = " << width << "; height = " << height << std::endl;

        glActiveTexture(GL_TEXTURE0 + point_shadow_texture_unit);
        glBindTexture(GL_TEXTURE_CUBE_MAP, depthCubemap);

        glUniform3f(point_light_position_location, point_light_position.x, point_light_position.y, point_light_position.z);
        glUniform3f(point_light_color_location, 0.f, 1.f, 1.f);
        glUniform3f(point_light_attenuation_location, 0.01f, 0.001f, 0.0001f);
        glUniform1i(depthMap_location, point_shadow_texture_unit);
        glUniform1f(far_plane_location, far2);

        //std::cout << "i = " << i++ << "; width = " << width << "; height = " << height << std::endl;


        first = 0;
        for (size_t s = 0; s < shapes.size(); s++) {
            auto pupupu = convertToSet(shapes[s].mesh.material_ids);
            if (pupupu.size() == 2) {
                // std::cout << shapes[s].name << std::endl;
                // std::cout << "s = " << s << std::endl;
                // for (auto i: pupupu) {
                //     std::cout << "i = " << i << std::endl;
                //     std::cout << "materials[id].ambient_texname = " << materials[i].ambient_texname << std::endl;
                // }
                int flag;
                for (int i = 1; i < shapes[s].mesh.material_ids.size(); i += 1) {
                    if (shapes[s].mesh.material_ids[i] != shapes[s].mesh.material_ids[i - 1]) {
                        flag = i;
                        // std::cout << i << std::endl;
                        break;
                        // std::cout << "materials[id].ambient_texname = " << materials[shapes[s].mesh.material_ids[i]].ambient_texname << std::endl;
                        // std::cout << "materials[id].ambient_texname = " << materials[shapes[s].mesh.material_ids[i-1]].ambient_texname << std::endl;
                        // std::cout << std::endl;
                        // // std::cout << "num_face_vertices[id] = " << shapes[1].mesh.num_face_vertices[i] << std::endl;
                    }
                }

                int id = shapes[s].mesh.material_ids[flag-1];
                // std::cout << "materials[id].ambient_texname 1 = " << materials[id].ambient_texname << std::endl;
                // std::string texture_path;
                // if (materials[id].ambient_texname == "textures/vase_plant.png") {
                //     texture_path = "/Users/maria.barkovskaya/Documents/uni/comp_graph/graphics-course-practice/hw2/textures/vase_round.png";
                //     std::cout << "vase_plant" << std::endl;
                // } else {
                //     texture_path = materials_dir + materials[id].ambient_texname;
                // }
                std::string texture_path = materials_dir + materials[id].ambient_texname;
                std::replace(texture_path.begin(), texture_path.end(), '\\', '/');

                // if (texture_path == "/Users/maria.barkovskaya/Documents/uni/comp_graph/graphics-course-practice/hw2/textures/vase_round.png") {
                //     std::cout << "materials[id].ambient_texnam = " << materials[id].ambient_texname << std::endl;
                //     // glActiveTexture(GL_TEXTURE0);
                //     // glUniform1i(ambient_texture_location, 0);
                //     // continue;
                // } else {
                //     glUniform1i(ambient_texture_location, textures.get_texture(texture_path));
                // }

                glUniform1i(ambient_texture_location, textures.get_texture(texture_path));
                glUniform1f(glossiness_location, materials[id].specular[0]);
                glUniform1f(power_location, materials[id].shininess);

                texture_path = materials_dir + materials[id].alpha_texname;
                std::replace(texture_path.begin(), texture_path.end(), '\\', '/');

                // if (texture_path == "/Users/maria.barkovskaya/Documents/uni/comp_graph/graphics-course-practice/hw2/textures/vase_plant_mask.png") {
                //         std::cout << "1: vase; materials[id].alpha_texname.empty() = " << materials[id].alpha_texname.empty() << std::endl;
                //     }

                glUniform1i(alpha_texture_location, textures.get_texture(texture_path));
                glUniform1i(flag_map_d_location, !materials[id].alpha_texname.empty());

                glDrawArrays(GL_TRIANGLES, first, (GLint)(flag*3));
                first += flag * 3;

                id = shapes[s].mesh.material_ids[flag];
                // std::cout << "materials[id].ambient_texname 2 = " << materials[id].ambient_texname << std::endl;
                // std::string texture_path;
                // if (materials[id].ambient_texname == "textures/vase_plant.png") {
                //     texture_path = "/Users/maria.barkovskaya/Documents/uni/comp_graph/graphics-course-practice/hw2/textures/vase_round.png";
                //     std::cout << "vase_plant" << std::endl;
                // } else {
                //     texture_path = materials_dir + materials[id].ambient_texname;
                // }
                texture_path = materials_dir + materials[id].ambient_texname;
                std::replace(texture_path.begin(), texture_path.end(), '\\', '/');

                // if (texture_path == "/Users/maria.barkovskaya/Documents/uni/comp_graph/graphics-course-practice/hw2/textures/vase_round.png") {
                //     std::cout << "materials[id].ambient_texnam = " << materials[id].ambient_texname << std::endl;
                //     // glActiveTexture(GL_TEXTURE0);
                //     // glUniform1i(ambient_texture_location, 0);
                //     // continue;
                // } else {
                //     glUniform1i(ambient_texture_location, textures.get_texture(texture_path));
                // }

                glUniform1i(ambient_texture_location, textures.get_texture(texture_path));
                glUniform1f(glossiness_location, materials[id].specular[0]);
                glUniform1f(power_location, materials[id].shininess);

                texture_path = materials_dir + materials[id].alpha_texname;
                std::replace(texture_path.begin(), texture_path.end(), '\\', '/');

                // if (texture_path == "/Users/maria.barkovskaya/Documents/uni/comp_graph/graphics-course-practice/hw2/textures/vase_plant_mask.png") {
                //         std::cout << "1: vase; materials[id].alpha_texname.empty() = " << materials[id].alpha_texname.empty() << std::endl;
                //     }

                glUniform1i(alpha_texture_location, textures.get_texture(texture_path));
                glUniform1i(flag_map_d_location, !materials[id].alpha_texname.empty());

                glDrawArrays(GL_TRIANGLES, first, (GLint)(shapes[s].mesh.indices.size() - flag * 3));
                first += (shapes[s].mesh.indices.size() - flag * 3);

                // std::cout << std::endl;
            } else if (pupupu.size() == 1) {
                int id = shapes[s].mesh.material_ids[0];
                // std::string texture_path;
                // if (materials[id].ambient_texname == "textures/vase_plant.png") {
                //     texture_path = "/Users/maria.barkovskaya/Documents/uni/comp_graph/graphics-course-practice/hw2/textures/vase_round.png";
                //     std::cout << "vase_plant" << std::endl;
                // } else {
                //     texture_path = materials_dir + materials[id].ambient_texname;
                // }
                std::string texture_path = materials_dir + materials[id].ambient_texname;
                std::replace(texture_path.begin(), texture_path.end(), '\\', '/');

                // if (texture_path == "/Users/maria.barkovskaya/Documents/uni/comp_graph/graphics-course-practice/hw2/textures/vase_round.png") {
                //     std::cout << "materials[id].ambient_texnam = " << materials[id].ambient_texname << std::endl;
                //     // glActiveTexture(GL_TEXTURE0);
                //     // glUniform1i(ambient_texture_location, 0);
                //     // continue;
                // } else {
                //     glUniform1i(ambient_texture_location, textures.get_texture(texture_path));
                // }

                glUniform1i(ambient_texture_location, textures.get_texture(texture_path));
                glUniform1f(glossiness_location, materials[id].specular[0]);
                glUniform1f(power_location, materials[id].shininess);

                texture_path = materials_dir + materials[id].alpha_texname;
                std::replace(texture_path.begin(), texture_path.end(), '\\', '/');

                // if (texture_path == "/Users/maria.barkovskaya/Documents/uni/comp_graph/graphics-course-practice/hw2/textures/vase_plant_mask.png") {
                //         std::cout << "1: vase; materials[id].alpha_texname.empty() = " << materials[id].alpha_texname.empty() << std::endl;
                //     }

                glUniform1i(alpha_texture_location, textures.get_texture(texture_path));
                glUniform1i(flag_map_d_location, !materials[id].alpha_texname.empty());

                glDrawArrays(GL_TRIANGLES, first, (GLint)shapes[s].mesh.indices.size());
                first += shapes[s].mesh.indices.size();
            } else {
                std::cout << "pupupu.size() = " << pupupu.size() << std::endl;
                return 0;
            }
        }

        //std::cout << "i = " << i++ << "; width = " << width << "; height = " << height << std::endl;
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