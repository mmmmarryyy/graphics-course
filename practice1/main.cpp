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

GLuint create_shader(GLenum shader_type, const char* shader_source) {
    GLuint shader_id = glCreateShader(shader_type);
    glShaderSource(shader_id, 1, &shader_source, NULL);
    glCompileShader(shader_id);
    GLint isCompiled = 0;
    glGetShaderiv(shader_id, GL_COMPILE_STATUS, &isCompiled);

    if(isCompiled == GL_FALSE)
    {
        GLsizei number_of_characters;
        glGetShaderiv(shader_id, GL_INFO_LOG_LENGTH, &number_of_characters);
        std::string log_info(number_of_characters, '\0');
        glGetShaderInfoLog(shader_id, 1024, &number_of_characters, log_info.data());
        throw std::runtime_error(to_string("runtime error: ") + log_info);
    }
    return shader_id;
}

GLuint create_program(GLuint vertex_shader, GLuint fragment_shader) {
    GLuint program = glCreateProgram();
    glAttachShader(program, vertex_shader);
    glAttachShader(program, fragment_shader);
    glLinkProgram(program);
    GLint isLinked = 0;
    glGetProgramiv(program, GL_LINK_STATUS, &isLinked);
    if (isLinked == GL_FALSE) {
        GLint maxLength = 0;
        glGetProgramiv(program, GL_INFO_LOG_LENGTH, &maxLength);
        std::string log_info(maxLength, '\0');
        glGetProgramInfoLog(program, maxLength, &maxLength, log_info.data());
        glDeleteProgram(program);
        throw std::runtime_error(to_string("runtime error: ") + log_info);
    }
    return program;
}

const char fragment_source[] =
R"(#version 330 core

layout (location = 0) out vec4 out_color;

flat in vec3 color;

void main() {
    // vec4(R,G,B,A)
    // out_color = vec4(color, 1.0);

    int x = int(gl_FragCoord.x);
    int y = int(gl_FragCoord.y);

    bool is_even = ((x/20 + y/20) % 2 == 0);

    if (is_even) {
        out_color = vec4(color, 1.0);
    } else {
        out_color = vec4(1 - color, 1.0);
    }
}
)";

const char vertex_source[] =
R"(#version 330 core

const vec2 VERTICES[3] = vec2[3](
    vec2(0.0, 0.0),
    vec2(1.0, 0.0),
    vec2(0.0, 1.0)
);

flat out vec3 color;

void main() {
    gl_Position = vec4(VERTICES[gl_VertexID], 0.0, 1.0);
    color = vec3(gl_VertexID, 1.0, 0.0);
}
)";

int main() try
{
    if (SDL_Init(SDL_INIT_VIDEO) != 0)
        sdl2_fail("SDL_Init: ");

    SDL_Window * window = SDL_CreateWindow("Graphics course practice 1",
        SDL_WINDOWPOS_CENTERED,
        SDL_WINDOWPOS_CENTERED,
        800, 600,
        SDL_WINDOW_OPENGL | SDL_WINDOW_RESIZABLE | SDL_WINDOW_MAXIMIZED);

    if (!window)
        sdl2_fail("SDL_CreateWindow: ");

    SDL_GL_SetAttribute(SDL_GL_CONTEXT_MAJOR_VERSION, 3);
    SDL_GL_SetAttribute(SDL_GL_CONTEXT_MINOR_VERSION, 3);
    SDL_GL_SetAttribute(SDL_GL_CONTEXT_PROFILE_MASK, SDL_GL_CONTEXT_PROFILE_CORE);
    SDL_GL_SetAttribute(SDL_GL_DOUBLEBUFFER, 1);

    SDL_GLContext gl_context = SDL_GL_CreateContext(window);
    if (!gl_context)
        sdl2_fail("SDL_GL_CreateContext: ");

    if (auto result = glewInit(); result != GLEW_NO_ERROR)
        glew_fail("glewInit: ", result);

    if (!GLEW_VERSION_3_3)
        throw std::runtime_error("OpenGL 3.3 is not supported");

    glClearColor(0.8f, 0.8f, 1.f, 0.f);

    // GLenum shader_type = GL_FRAGMENT_SHADER;
    // create_shader(shader_type, "Hello world");

    GLenum shader_type = GL_FRAGMENT_SHADER;
    GLuint fragment_shader = create_shader(shader_type, fragment_source);

    GLenum shader_type2 = GL_VERTEX_SHADER;
    GLuint vertex_shader = create_shader(shader_type2, vertex_source);

    GLuint program_id = create_program(vertex_shader, fragment_shader);

    GLuint vao[1];
    glGenVertexArrays(1, vao);

    bool running = true;
    while (running)
    {
        for (SDL_Event event; SDL_PollEvent(&event);) switch (event.type)
        {
        case SDL_QUIT:
            running = false;
            break;
        }

        if (!running)
            break;

        glClear(GL_COLOR_BUFFER_BIT);

        glUseProgram(program_id);
        glBindVertexArray(vao[0]);
        glDrawArrays(GL_TRIANGLES, 0, 3);

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
