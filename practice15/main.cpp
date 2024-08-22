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

#include <glm/vec3.hpp>
#include <glm/mat4x4.hpp>
#include <glm/ext/matrix_transform.hpp>
#include <glm/ext/matrix_clip_space.hpp>
#include <glm/ext/scalar_constants.hpp>
#include <glm/gtx/quaternion.hpp>
#include <glm/gtx/string_cast.hpp>

#include "msdf_loader.hpp"
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

const char msdf_vertex_shader_source[] =
R"(#version 330 core

layout (location = 0) in vec2 position;
layout (location = 1) in vec2 in_texcoord;

uniform mat4 transform;
out vec2 texcoord;

void main()
{
    gl_Position = transform * vec4(position, 0.0, 1.0);
    texcoord = in_texcoord;
}
)";

const char msdf_fragment_shader_source[] =
R"(#version 330 core

layout (location = 0) out vec4 out_color;

uniform sampler2D sdf_texture;
uniform float sdf_scale;

in vec2 texcoord;

float median(vec3 v) {
    return max(min(v.r, v.g), min(max(v.r, v.g), v.b));
}

void main()
{
    float sdf_value = sdf_scale * (median(texture(sdf_texture, texcoord).rgb) - 0.5);
    float smooth_constant = length(vec2(dFdx(sdf_value), dFdy(sdf_value)))/sqrt(2.0);
    float alpha = smoothstep(-smooth_constant, smooth_constant, sdf_value);
    vec3 color = vec3(0.0);
    if (sdf_value < 0.4) {
        color = vec3(1.0);
    }
    out_color = vec4(color, alpha);
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

struct vertex {
    glm::vec2 position;
    glm::vec2 texcoord;

    vertex(glm::vec2 position, glm::vec2 texcoord) {
        this->position = position;
        this->texcoord = texcoord;
    }
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

    SDL_Window * window = SDL_CreateWindow("Graphics course practice 15",
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

    auto msdf_vertex_shader = create_shader(GL_VERTEX_SHADER, msdf_vertex_shader_source);
    auto msdf_fragment_shader = create_shader(GL_FRAGMENT_SHADER, msdf_fragment_shader_source);
    auto msdf_program = create_program(msdf_vertex_shader, msdf_fragment_shader);

    GLuint transform_location = glGetUniformLocation(msdf_program, "transform");
    GLuint scale_location = glGetUniformLocation(msdf_program, "sdf_scale");

    const std::string project_root = PROJECT_ROOT;
    const std::string font_path = project_root + "/font/font-msdf.json";

    auto const font = load_msdf_font(font_path);

    std::vector<vertex> vertices;
    vertices.push_back(vertex({0, 0}, {0, 0}));
    vertices.push_back(vertex({100, 0}, {1, 0}));
    vertices.push_back(vertex({0, 100}, {0, 1}));

    GLuint vao;
    glGenVertexArrays(1, &vao);
    glBindVertexArray(vao);

    GLuint vbo;
    glGenBuffers(1, &vbo);
    glBindBuffer(GL_ARRAY_BUFFER, vbo);
    // glBufferData(GL_ARRAY_BUFFER, vertices.size() * sizeof(vertices[0]), vertices.data(), GL_STATIC_DRAW);

    glEnableVertexAttribArray(0);
    glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, sizeof(vertex), (void *) offsetof(vertex, position));
    glEnableVertexAttribArray(1);
    glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, sizeof(vertex), (void *) offsetof(vertex, texcoord));

    GLuint texture;
    int texture_width, texture_height;
    {
        int channels;
        auto data = stbi_load(font.texture_path.c_str(), &texture_width, &texture_height, &channels, 4);
        assert(data);

        glGenTextures(1, &texture);
        glBindTexture(GL_TEXTURE_2D, texture);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR);
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, texture_width, texture_height, 0, GL_RGBA, GL_UNSIGNED_BYTE, data);
        glGenerateMipmap(GL_TEXTURE_2D);

        stbi_image_free(data);
    }

    auto last_frame_start = std::chrono::high_resolution_clock::now();

    float time = 0.f;

    SDL_StartTextInput();

    std::map<SDL_Keycode, bool> button_down;

    std::string text = "Hello, world!";
    bool text_changed = true;

    glm::vec3 bounding_box(0.f);

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
            if (event.key.keysym.sym == SDLK_BACKSPACE && !text.empty())
            {
                text.pop_back();
                text_changed = true;
            }
            break;
        case SDL_TEXTINPUT:
            text.append(event.text.text);
            text_changed = true;
        case SDL_KEYUP:
            button_down[event.key.keysym.sym] = false;
            break;
        }

        if (!running)
            break;

        auto now = std::chrono::high_resolution_clock::now();
        float dt = std::chrono::duration_cast<std::chrono::duration<float>>(now - last_frame_start).count();
        last_frame_start = now;

        glClearColor(0.8f, 0.8f, 1.f, 0.f);
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

        glEnable(GL_BLEND);
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

        glActiveTexture(GL_TEXTURE0);
        glBindTexture(GL_TEXTURE_2D, texture);

        if (text_changed) {
            glm::vec2 pen(0.f);
            vertices.clear();
            for (char c: text) {
                auto glyph = font.glyphs.at(c);

                glm::vec2 texture_sizes = {texture_width, texture_height};

                auto vertex_1 = vertex({glyph.xoffset + pen.x, glyph.yoffset + pen.y}, (glm::vec2{glyph.x, glyph.y} / texture_sizes));
                auto vertex_2 = vertex({glyph.xoffset + pen.x, glyph.yoffset + glyph.height + pen.y}, (glm::vec2{glyph.x, glyph.y + glyph.height} / texture_sizes));
                auto vertex_3 = vertex({glyph.xoffset + glyph.width + pen.x, glyph.yoffset + pen.y}, (glm::vec2{glyph.x + glyph.width, glyph.y} / texture_sizes));
                auto vertex_4 = vertex({glyph.xoffset + glyph.width + pen.x, glyph.yoffset + glyph.height + pen.y}, (glm::vec2{glyph.x + glyph.width, glyph.y + glyph.height} / texture_sizes));
                
                vertices.push_back(vertex_1);
                vertices.push_back(vertex_2);
                vertices.push_back(vertex_3);
                vertices.push_back(vertex_3);
                vertices.push_back(vertex_2);
                vertices.push_back(vertex_4);

                pen += glm::vec2{glyph.advance, 0.f};

                bounding_box.x = -pen.x / 2;
                bounding_box.y = fmin(bounding_box.y, -(float)glyph.height);
            }
            text_changed = false;
            glBufferData(GL_ARRAY_BUFFER, vertices.size() * sizeof(vertices[0]), vertices.data(), GL_STATIC_DRAW);
        }

        glm::mat4 transform(1.f);
        transform = glm::scale(transform, glm::vec3({2.f / width, -2.f / height, 0.f}));
        transform = glm::scale(transform, glm::vec3(5.f));
        transform = glm::translate(transform, bounding_box);

        glUseProgram(msdf_program);
        glUniformMatrix4fv(transform_location, 1, GL_FALSE, reinterpret_cast<float *>(&transform));
        glUniform1f(scale_location, font.sdf_scale);
        glBindVertexArray(vao);

        glDrawArrays(GL_TRIANGLES, 0, vertices.size());

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
