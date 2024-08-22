#ifdef WIN32
#include <SDL.h>
#undef main
#else
#include <SDL2/SDL.h>
#endif

#include <GL/glew.h>

#include <string_view>
#include <stdexcept>
#include <iostream>
#include <fstream>
#include <chrono>
#include <vector>
#include <random>
#include <map>
#include <cmath>

#define GLM_FORCE_SWIZZLE
#define GLM_ENABLE_EXPERIMENTAL
#include <glm/vec3.hpp>
#include <glm/mat4x4.hpp>
#include <glm/ext/matrix_transform.hpp>
#include <glm/ext/matrix_clip_space.hpp>
#include <glm/ext/scalar_constants.hpp>
#include <glm/gtx/quaternion.hpp>
#include <glm/gtx/string_cast.hpp>

#include "obj_parser.hpp"
#include "stb_image.h"

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

const char vertex_shader_source[] =
R"(#version 330 core

uniform mat4 view;
uniform mat4 projection;

uniform vec3 bbox_min;
uniform vec3 bbox_max;

layout (location = 0) in vec3 in_position;

out vec3 position;

void main()
{
    position = bbox_min + in_position * (bbox_max - bbox_min);
    gl_Position = projection * view * vec4(position, 1.0);
}
)";

const char fragment_shader_source[] =
R"(#version 330 core

uniform vec3 camera_position;
uniform vec3 light_direction;
uniform vec3 bbox_min;
uniform vec3 bbox_max;

uniform sampler3D cloud_data;

layout (location = 0) out vec4 out_color;

void sort(inout float x, inout float y)
{
    if (x > y)
    {
        float t = x;
        x = y;
        y = t;
    }
}

float vmin(vec3 v)
{
    return min(v.x, min(v.y, v.z));
}

float vmax(vec3 v)
{
    return max(v.x, max(v.y, v.z));
}

vec2 intersect_bbox(vec3 origin, vec3 direction)
{
    vec3 tmin = (bbox_min - origin) / direction;
    vec3 tmax = (bbox_max - origin) / direction;

    sort(tmin.x, tmax.x);
    sort(tmin.y, tmax.y);
    sort(tmin.z, tmax.z);

    return vec2(vmax(tmin), vmin(tmax));
}

float texture_value(vec3 point) 
{
    vec3 fixed_point = (point - bbox_min) / (bbox_max - bbox_min);
    return texture(cloud_data, fixed_point).x;
}

const float PI = 3.1415926535;
float absorption = 0.0;

in vec3 position;

void main()
{
    vec3 direction = normalize(position - camera_position);
    vec2 intersect_interval = intersect_bbox(camera_position, direction);
    float tmax = intersect_interval.y;
    float tmin = intersect_interval.x;
    tmin = max(tmin, 0.0);

    //////////////////////////////////////////////////////////////////////

    float optical_depth = (tmax - tmin) * absorption;
    float opacity = 1.0 - exp(-optical_depth);

    //////////////////////////////////////////////////////////////////////

    vec3 point = camera_position + direction * (tmin + tmax) / 2;

    //////////////////////////////////////////////////////////////////////

    optical_depth = 0.0;
    int n = 64;
    float dt = (tmax - tmin) / n;
    for (int i = 0; i < n; i++) {
        float t = tmin + (i + 0.5) * dt;
        point = camera_position + t * direction;
        optical_depth += absorption * texture_value(point) * dt;
    }
    opacity = 1.0 - exp(-optical_depth);

    //////////////////////////////////////////////////////////////////////
    
    float scattering = 4.0;
    float extinction = absorption + scattering;
    vec3 light_color = vec3(16.0);
    vec3 color = vec3(0.0);

    n = 64; 
    int m = 16;

    dt = (tmax - tmin) / n;
    optical_depth = 0.0;

    for (int i = 0; i < n; i++) {
        float t = tmin + (i + 0.5) * dt;
        point = camera_position + t * direction;
        optical_depth += extinction * texture_value(point) * dt;

        vec2 light_tmin_tmax = intersect_bbox(point, light_direction);
        float light_tmin = light_tmin_tmax.x;
        float light_tmax = light_tmin_tmax.y;
        light_tmin = max(light_tmin, 0.0);

        float dl = (light_tmax - light_tmin) / m;

        float light_optical_depth = 0.0;

        for (int j = 0; j < m; j++) {
            float l = light_tmin + (j + 0.5) * dl;
            vec3 l_point = point + l * light_direction;

            light_optical_depth += extinction * texture_value(l_point) * dl;
        }

        color += light_color * exp(-light_optical_depth) * exp(-optical_depth) * dt * texture_value(point) * scattering / 4.0 / PI;
    }

    opacity = 1.0 - exp(-optical_depth);

    //////////////////////////////////////////////////////////////////////
    
    vec3 vec_scattering = vec3(1.0, 5.8, 9.2);
    vec3 vec_absorption = vec3(absorption);
    vec3 vec_extinction = vec_absorption + vec_scattering;

    vec3 vec_light_color = vec3(16.0);
    color = vec3(0.0);

    vec3 vec_optical_depth = vec3(0.0);

    for (int i = 0; i < n; i++) {
        float t = tmin + (i + 0.5) * dt;
        point = camera_position + t * direction;
        vec_optical_depth += vec_extinction * texture_value(point) * dt;

        vec3 vec_light_optical_depth = vec3(0.0);
        vec2 light_tmin_tmax = intersect_bbox(point, light_direction);
        float light_tmin = light_tmin_tmax.x;
        float light_tmax = light_tmin_tmax.y;
        light_tmin = max(light_tmin, 0.0);

        float dl = (light_tmax - light_tmin) / m;

        for (int j = 0; j < m; j++) {
            float l = light_tmin + (j + 0.5) * dl;
            vec3 l_point = point + l * light_direction;

            vec_light_optical_depth += vec_extinction * texture_value(l_point) * dl;
        }

        color += vec_light_color * exp(-vec_light_optical_depth) * exp(-vec_optical_depth) * dt * texture_value(point) * vec_scattering / 4.0 / PI;
    }
    opacity = 1.0 - exp(-vec_optical_depth.x);


    // out_color = vec4(vec3((tmax - tmin)/4), 1.0);
    // out_color = vec4(0.0, 0.0, 0.5, opacity);
    // out_color = vec4(vec3(texture_value(point)), 1.0);
    out_color = vec4(color, opacity);
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

template <typename ... Shaders>
GLuint create_program(Shaders ... shaders)
{
    GLuint result = glCreateProgram();
    (glAttachShader(result, shaders), ...);
    glLinkProgram(result);

    GLint status;
    glGetProgramiv(result, GL_LINK_STATUS, &status);
    if (status != GL_TRUE)
    {
        GLint info_log_length;
        glGetProgramiv(result, GL_INFO_LOG_LENGTH, &info_log_length);
        std::string info_log(info_log_length, '\0');
        glGetProgramInfoLog(result, info_log.size(), nullptr, info_log.data());
        throw std::runtime_error("Program linkage failed: " + info_log);
    }

    return result;
}

static glm::vec3 cube_vertices[]
{
    {0.f, 0.f, 0.f},
    {1.f, 0.f, 0.f},
    {0.f, 1.f, 0.f},
    {1.f, 1.f, 0.f},
    {0.f, 0.f, 1.f},
    {1.f, 0.f, 1.f},
    {0.f, 1.f, 1.f},
    {1.f, 1.f, 1.f},
};

static std::uint32_t cube_indices[]
{
	// -Z
	0, 2, 1,
	1, 2, 3,
	// +Z
	4, 5, 6,
	6, 5, 7,
	// -Y
	0, 1, 4,
	4, 1, 5,
	// +Y
	2, 6, 3,
	3, 6, 7,
	// -X
	0, 4, 2,
	2, 4, 6,
	// +X
	1, 3, 5,
	5, 3, 7,
};

int main() try
{
    if (SDL_Init(SDL_INIT_VIDEO) != 0)
        sdl2_fail("SDL_Init: ");

    SDL_GL_SetAttribute(SDL_GL_CONTEXT_MAJOR_VERSION, 3);
    SDL_GL_SetAttribute(SDL_GL_CONTEXT_MINOR_VERSION, 3);
    SDL_GL_SetAttribute(SDL_GL_CONTEXT_PROFILE_MASK, SDL_GL_CONTEXT_PROFILE_CORE);
    SDL_GL_SetAttribute(SDL_GL_DOUBLEBUFFER, 1);
    SDL_GL_SetAttribute(SDL_GL_RED_SIZE, 8);
    SDL_GL_SetAttribute(SDL_GL_GREEN_SIZE, 8);
    SDL_GL_SetAttribute(SDL_GL_BLUE_SIZE, 8);
    SDL_GL_SetAttribute(SDL_GL_DEPTH_SIZE, 24);

    SDL_Window * window = SDL_CreateWindow("Graphics course practice 11",
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

    auto vertex_shader = create_shader(GL_VERTEX_SHADER, vertex_shader_source);
    auto fragment_shader = create_shader(GL_FRAGMENT_SHADER, fragment_shader_source);
    auto program = create_program(vertex_shader, fragment_shader);

    GLuint view_location = glGetUniformLocation(program, "view");
    GLuint projection_location = glGetUniformLocation(program, "projection");
    GLuint bbox_min_location = glGetUniformLocation(program, "bbox_min");
    GLuint bbox_max_location = glGetUniformLocation(program, "bbox_max");
    GLuint camera_position_location = glGetUniformLocation(program, "camera_position");
    GLuint light_direction_location = glGetUniformLocation(program, "light_direction");
    GLuint cloud_texture_location = glGetUniformLocation(program, "cloud_data");

    GLuint vao, vbo, ebo;
    glGenVertexArrays(1, &vao);
    glBindVertexArray(vao);

    glGenBuffers(1, &vbo);
    glBindBuffer(GL_ARRAY_BUFFER, vbo);
    glBufferData(GL_ARRAY_BUFFER, sizeof(cube_vertices), cube_vertices, GL_STATIC_DRAW);

    glGenBuffers(1, &ebo);
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, ebo);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(cube_indices), cube_indices, GL_STATIC_DRAW);

    glEnableVertexAttribArray(0);
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 0, nullptr);

    const std::string project_root = PROJECT_ROOT;
    const std::string cloud_data_path = project_root + "/cloud.data";

    GLuint cloud_texture;
    glGenTextures(1, &cloud_texture);
    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_3D, cloud_texture);
    glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_WRAP_R, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    std::vector<char> pixels(128 * 64 * 64); //x * y * z
    std::ifstream input(cloud_data_path, std::ios::binary);
    input.read(pixels.data(), pixels.size());
    glTexImage3D(GL_TEXTURE_3D, 0, GL_R8, 128, 64, 64, 0, GL_RED, GL_UNSIGNED_BYTE, pixels.data());

    const glm::vec3 cloud_bbox_min{-2.f, -1.f, -1.f};
    const glm::vec3 cloud_bbox_max{ 2.f,  1.f,  1.f};

    auto last_frame_start = std::chrono::high_resolution_clock::now();

    float time = 0.f;

    std::map<SDL_Keycode, bool> button_down;

    float view_angle = glm::pi<float>() / 6.f;
    float camera_distance = 3.5f;

    float camera_rotation = glm::pi<float>() / 6.f;

    bool paused = false;

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

        if (button_down[SDLK_UP])
            camera_distance -= 3.f * dt;
        if (button_down[SDLK_DOWN])
            camera_distance += 3.f * dt;

        if (button_down[SDLK_a])
            camera_rotation -= 2.f * dt;
        if (button_down[SDLK_d])
            camera_rotation += 2.f * dt;

        if (button_down[SDLK_w])
            view_angle -= 2.f * dt;
        if (button_down[SDLK_s])
            view_angle += 2.f * dt;

        glClearColor(0.0f, 0.0f, 0.0f, 0.f);
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

        glEnable(GL_DEPTH_TEST);

        glEnable(GL_CULL_FACE);
        glCullFace(GL_FRONT);

        glEnable(GL_BLEND);
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

        float near = 0.1f;
        float far = 100.f;

        glm::mat4 model(1.f);

        glm::mat4 view(1.f);
        view = glm::translate(view, {0.f, 0.f, -camera_distance});
        view = glm::rotate(view, view_angle, {1.f, 0.f, 0.f});
        view = glm::rotate(view, camera_rotation, {0.f, 1.f, 0.f});

        glm::mat4 projection = glm::perspective(glm::pi<float>() / 2.f, (1.f * width) / height, near, far);

        glm::vec3 camera_position = (glm::inverse(view) * glm::vec4(0.f, 0.f, 0.f, 1.f)).xyz();

        glm::vec3 light_direction = glm::normalize(glm::vec3(std::cos(time), 1.f, std::sin(time)));

        glUseProgram(program);
        glUniformMatrix4fv(view_location, 1, GL_FALSE, reinterpret_cast<float *>(&view));
        glUniformMatrix4fv(projection_location, 1, GL_FALSE, reinterpret_cast<float *>(&projection));
        glUniform3fv(bbox_min_location, 1, reinterpret_cast<const float *>(&cloud_bbox_min));
        glUniform3fv(bbox_max_location, 1, reinterpret_cast<const float *>(&cloud_bbox_max));
        glUniform3fv(camera_position_location, 1, reinterpret_cast<float *>(&camera_position));
        glUniform3fv(light_direction_location, 1, reinterpret_cast<float *>(&light_direction));
        glUniform1i(cloud_texture_location, 0);

        glBindVertexArray(vao);
        glDrawElements(GL_TRIANGLES, std::size(cube_indices), GL_UNSIGNED_INT, nullptr);

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