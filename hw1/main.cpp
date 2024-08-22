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
#include <chrono>
#include <vector>
#include <map>
#include <random>

const float PI = acos(-1);

std::string to_string(std::string_view str) {
	return std::string(str.begin(), str.end());
}

void sdl2_fail(std::string_view message) {
	throw std::runtime_error(to_string(message) + SDL_GetError());
}

void glew_fail(std::string_view message, GLenum error) {
	throw std::runtime_error(to_string(message) + reinterpret_cast<const char *>(glewGetErrorString(error)));
}

const char vertex_shader_source[] =
R"(#version 330 core

uniform mat4 view;
uniform mat4 rotate_xz;
uniform mat4 rotate_yz;
uniform mat4 rotate_xy;

layout (location = 0) in vec2 in_position;
layout (location = 1) in float height;
layout (location = 2) in vec4 in_color;

out vec4 color;

void main() {
	gl_Position =  rotate_xz * rotate_yz * rotate_xy * view * vec4(in_position, height, 1.0);
	color = in_color;
}
)";

const char fragment_shader_source[] =
R"(#version 330 core

in vec4 color;

layout (location = 0) out vec4 out_color;

void main() {
	out_color = color;
}
)";

GLuint create_shader(GLenum type, const char * source) {
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

GLuint create_program(GLuint vertex_shader, GLuint fragment_shader) {
    GLuint result = glCreateProgram();
    glAttachShader(result, vertex_shader);
    glAttachShader(result, fragment_shader);
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

struct vec2 {
	float x;
	float y;
};

struct color {
	std::uint8_t color[4];
};

struct vertex {
    vec2 position;
	float height;
    color color;
};

struct metaball {
	vec2 position;
	vec2 direction;
	float r;
	float c;
};

GLuint VAO, VBO_P, VBO_H, VBO_C, EBO, VAO2, VBO2, EBO2;

int grid_size = 100;
float number_of_isolines = 4;
float max_z, min_z;

std::vector<vertex> grid_vertices;
std::vector<int> grid_indices;
std::vector<vertex> isoline_vertices;
std::vector<int> isoline_indices;
// std::map<std::pair<int, int>, int> indices_after_interpolation;

std::random_device rd;
std::mt19937 seed(rd());
std::uniform_real_distribution<> dist_pos(-4.f, 4.f);
std::uniform_real_distribution<> dist_dir(-2.f, 2.f);
std::uniform_real_distribution<> dist_r(0.9f, 1.9f);
std::uniform_real_distribution<> dist_c(-1.8f, 1.8f);

std::vector<metaball> metaballs = {
	{{0.f, 0.5f}, {1.f, -0.3f}, 1.3f, 1.2f},
	{{0.6f, 0.1f}, {-0.5f, -0.7f}, 1.2f, 0.9f},
	{{1.f, 0.f}, {0.6f, -0.4f}, 1.5f, -1.3f},
	{{1.f, -0.7f}, {-1.f, 0.7f}, 0.8f, 1.5f},
	{{0.f, 0.f}, {-0.9f, 0.5f}, 1.5f, 0.5f},
	{{-1.f, 0.f}, {-1.f, 0.5f}, 0.9f, 1.5f}
};

void update_grid_vertices_buffer() {
	std::vector<vec2> positions;
	for (vertex vertex: grid_vertices) {
		positions.push_back(vertex.position);
	}

	glBindBuffer(GL_ARRAY_BUFFER, VBO_P);
	glBufferData(GL_ARRAY_BUFFER, positions.size() * sizeof(vertex::position), positions.data(), GL_DYNAMIC_DRAW);

	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, EBO);
	glBufferData(GL_ELEMENT_ARRAY_BUFFER, grid_indices.size() * sizeof(int), grid_indices.data(), GL_DYNAMIC_DRAW);
}

void update_grid_vertices_hc_buffer() {
	std::vector<float> heights;
	std::vector<color> colors;

	for (vertex vertex: grid_vertices) {
		heights.push_back(vertex.height);
		colors.push_back(vertex.color);
	}

	glBindBuffer(GL_ARRAY_BUFFER, VBO_H);
	glBufferData(GL_ARRAY_BUFFER, heights.size() * sizeof(vertex::height), heights.data(), GL_DYNAMIC_DRAW);
	glBindBuffer(GL_ARRAY_BUFFER, VBO_C);
	glBufferData(GL_ARRAY_BUFFER, colors.size() * sizeof(vertex::color), colors.data(), GL_DYNAMIC_DRAW);
}

void update_isolines_buffers() {
	glBindBuffer(GL_ARRAY_BUFFER, VBO2);
	glBufferData(GL_ARRAY_BUFFER, isoline_vertices.size() * sizeof(vertex), isoline_vertices.data(), GL_DYNAMIC_DRAW);
	
	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, EBO2);
	glBufferData(GL_ELEMENT_ARRAY_BUFFER, isoline_indices.size()*sizeof(int), isoline_indices.data(), GL_DYNAMIC_DRAW);
}

float metaball_function(float x, float y) {
	float result = 0.0;
	for (int i = 0; i < metaballs.size(); i++) {
		float x_i = metaballs[i].position.x;
		float y_i = metaballs[i].position.y;
		float c_i = metaballs[i].c;
		float r_i = metaballs[i].r;

		result += c_i * exp(-((x - x_i) * (x - x_i) + (y - y_i) * (y - y_i)) / r_i / r_i);
	}

	return result;
}

//https://habr.com/ru/articles/589753/
vertex interpolate_coords(int first_index, int second_index, float value) {
	vertex vertex1 = grid_vertices[first_index];
	vertex vertex2 = grid_vertices[second_index];

	if (vertex1.position.y == vertex2.position.y) {
		return vertex{
			vec2({vertex1.position.x + (value - vertex1.height) / (vertex2.height - vertex1.height) * (vertex2.position.x - vertex1.position.x), vertex1.position.y}), value + 0.01f, {255, 255, 255, 255}
			// vec2({vertex1.position.x + (value - vertex1.height) / (vertex2.height - vertex1.height) * (vertex2.position.x - vertex1.position.x), vertex1.position.y}), 0.01f, {255, 255, 255, 255}
		};
	} else {
		return vertex{
			vec2({vertex1.position.x, vertex1.position.y + (value - vertex1.height) / (vertex2.height - vertex1.height) * (vertex2.position.y - vertex1.position.y)}), value + 0.01f, {255, 255, 255, 255}
			// vec2({vertex1.position.x, vertex1.position.y + (value - vertex1.height) / (vertex2.height - vertex1.height) * (vertex2.position.y - vertex1.position.y)}), 0.01f, {255, 255, 255, 255}
		};
	}
}

void add_isoline(float isoline_height) {
	std::map<std::pair<int, int>, int> indices_after_interpolation;
	for (int i = 0; i < grid_size - 1; i++) {
		for (int j = 0; j < grid_size - 1; j++) {

			int grid_cell_ids[4] = {
				i * grid_size + j,     
				i * grid_size + j + 1,
				(i + 1) * grid_size + j + 1, 
				(i + 1) * grid_size + j
			};

			float grid_cells_height[4] = {
				grid_vertices[grid_cell_ids[0]].height - isoline_height,
				grid_vertices[grid_cell_ids[1]].height - isoline_height,
				grid_vertices[grid_cell_ids[2]].height - isoline_height,
				grid_vertices[grid_cell_ids[3]].height - isoline_height
			};

			std::vector<std::pair<int, int>> needs_interpolation;
			float mean = (grid_cells_height[0] + grid_cells_height[1] + grid_cells_height[2] + grid_cells_height[3]) / 4;
			int positive = 0;

			for (int k = 0; k < 4; k++) {
				positive += grid_cells_height[k] > 0;
			}

			switch (positive) {
				case 1:
				case 3:
					for (int k = 0; k < 4; k++) {
						if ((grid_cells_height[k] * grid_cells_height[(k + 3) % 4] < 0) && (grid_cells_height[k] * grid_cells_height[(k + 1) % 4] < 0)) {
							needs_interpolation.push_back(std::pair(k, (k + 3) % 4));
							needs_interpolation.push_back(std::pair(k, (k + 1) % 4));
						}
					}
					break;

				case 2:
					if (grid_cells_height[0] * grid_cells_height[1] > 0) {
						needs_interpolation.push_back(std::pair(0, 3));
						needs_interpolation.push_back(std::pair(1, 2));
					} else if (grid_cells_height[0] * grid_cells_height[3] > 0) {
						needs_interpolation.push_back(std::pair(0, 1));
						needs_interpolation.push_back(std::pair(2, 3));
					} else if (grid_cells_height[0] * mean > 0) {
						needs_interpolation.push_back(std::pair(0, 1));
						needs_interpolation.push_back(std::pair(1, 2));
						needs_interpolation.push_back(std::pair(0, 3));
						needs_interpolation.push_back(std::pair(2, 3));
					} else {
						needs_interpolation.push_back(std::pair(0, 1));
						needs_interpolation.push_back(std::pair(0, 3));
						needs_interpolation.push_back(std::pair(1, 2));
						needs_interpolation.push_back(std::pair(2, 3));
					}
					break;
				
				case 4:
				case 0:
				default:
					continue;
					break;
			}

			for (auto pair: needs_interpolation) {
				std::pair<int, int> indexes = std::pair(grid_cell_ids[pair.first], grid_cell_ids[pair.second]);
				if (indices_after_interpolation.contains(indexes)) {
					isoline_indices.push_back(indices_after_interpolation[indexes]);
				} else {
					isoline_vertices.push_back(interpolate_coords(indexes.first, indexes.second, isoline_height));
					indices_after_interpolation[indexes] = isoline_vertices.size() - 1;
					isoline_indices.push_back(isoline_vertices.size() - 1);
				}
			}
		}
	}
}

void update_isolines(float min_z, float max_z) {
	isoline_vertices.clear();
	isoline_indices.clear();

	for (int i = 1; i <= number_of_isolines; i++) {
		add_isoline(min_z + (max_z - min_z) * float(i) / (number_of_isolines + 1));
	}

	update_isolines_buffers();
}

void update_colors(float min_z, float max_z) {
	for (int i = 0; i < grid_size; i++) {
		for (int j = 0; j < grid_size; j++) {
			float c = (grid_vertices[i * grid_size + j].height - min_z) / (max_z - min_z);
			grid_vertices[i * grid_size + j].color = color({(uint8_t)(100 * c), (uint8_t)(150 * c), (uint8_t)(255 * c), 255});
		}
	}
}

void update_grid_vertices() {
	bool update_only_colors = (grid_vertices.size() == grid_size * grid_size);

	if (!update_only_colors) {
		grid_vertices.clear();
	}

	for (int i = 0; i < grid_size; i++) {
		for (int j = 0; j < grid_size; j++) {
			float x = 2.0f * i / (grid_size - 1) - 1;
			float y = 2.0f * j / (grid_size - 1) - 1;
			float z = metaball_function(5.0f * x, 5.0f * y) / 5.0f; //use 5.0f for making function values more stable

			if (i + j == 0) {
				max_z = z;
				min_z = z;
			} else {
				max_z = std::max(z, max_z);
				min_z = std::min(z, min_z);
			}

			if (update_only_colors) {
				grid_vertices[i * grid_size + j].height = z;
			} else {
				grid_vertices.push_back(vertex({vec2({x, -y}), z, color({0, 0, 0, 255})}));
			}
		}
	}

	update_colors(min_z, max_z);
	update_isolines(min_z, max_z);

	// for (int i = 0; i < grid_size; i++) {
	// 	for (int j = 0; j < grid_size; j++) {
	// 		grid_vertices[i * grid_size + j].height = 0;
	// 	}
	// }

	if (!update_only_colors) {
		update_grid_vertices_buffer();
	}
	update_grid_vertices_hc_buffer();
}

void update_grid_indices() {
	grid_indices.clear();

	for (int i = 0; i < grid_size - 1; i++) {
		for (int j = 0; j < grid_size - 1; j++) {
			grid_indices.push_back(i * grid_size + j + 1);
			grid_indices.push_back(i * grid_size + j + grid_size);
			grid_indices.push_back(i * grid_size + j);
			grid_indices.push_back(i * grid_size + j + 1);
			grid_indices.push_back(i * grid_size + j + 1 + grid_size);
			grid_indices.push_back(i * grid_size + j + grid_size);
		}
	}

	update_grid_vertices();
}

void update_metaballs(float dt) {
	for (int i = 0; i < metaballs.size(); i++) {
		metaballs[i].position.x += dt * metaballs[i].direction.x;
		metaballs[i].position.y += dt * metaballs[i].direction.y;
		if (abs(metaballs[i].position.x) > 5.0f) { //constant from metaball function stabilization
			metaballs[i].direction.x *= -1;
		}
		if (abs(metaballs[i].position.y) > 5.0f) {
			metaballs[i].direction.y *= -1;
		}
	}

	update_grid_vertices();
}

void add_random_metaballs(float count) {
	for (int i = 0; i < count; i++) {
		metaballs.push_back( metaball({
			vec2({static_cast<float>(dist_pos(seed)), static_cast<float>(dist_pos(seed))}),
			vec2({static_cast<float>(dist_dir(seed)), static_cast<float>(dist_dir(seed))}),
			static_cast<float>(dist_r(seed)),
			static_cast<float>(dist_c(seed))
		}));
	}
}

int main() try {
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

    SDL_Window * window = SDL_CreateWindow("Graphics course practice 4",
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

	glGenVertexArrays(1, &VAO);
	glBindVertexArray(VAO);

	glGenBuffers(1, &VBO_P);
    glBindBuffer(GL_ARRAY_BUFFER, VBO_P);

	glEnableVertexAttribArray(0);
	glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 0, (void*)(0));

	glGenBuffers(1, &VBO_H);
    glBindBuffer(GL_ARRAY_BUFFER, VBO_H);

	glEnableVertexAttribArray(1);
	glVertexAttribPointer(1, 1, GL_FLOAT, GL_FALSE, 0, (void*)(0));

	glGenBuffers(1, &VBO_C);
    glBindBuffer(GL_ARRAY_BUFFER, VBO_C);

	glEnableVertexAttribArray(2);
	glVertexAttribPointer(2, 4, GL_UNSIGNED_BYTE, GL_TRUE, 0, (void*)(0));

	glGenBuffers(1, &EBO);
	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, EBO);

	glGenBuffers(1, &VBO2);
    glBindBuffer(GL_ARRAY_BUFFER, VBO2);

	glGenVertexArrays(1, &VAO2);
	glBindVertexArray(VAO2);

	glEnableVertexAttribArray(0);
	glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, sizeof(vertex), reinterpret_cast<void*>(offsetof(vertex, position)));

	glEnableVertexAttribArray(1);
	glVertexAttribPointer(1, 1, GL_FLOAT, GL_FALSE, sizeof(vertex), reinterpret_cast<void*>(offsetof(vertex, height)));

	glEnableVertexAttribArray(2);
	glVertexAttribPointer(2, 4, GL_UNSIGNED_BYTE, GL_TRUE, sizeof(vertex), reinterpret_cast<void*>(offsetof(vertex, color)));

	glGenBuffers(1, &EBO2);
	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, EBO2);

	GLuint view_location = glGetUniformLocation(program, "view");
    GLuint rotate_location = glGetUniformLocation(program, "rotate_xz");
	GLuint rotate2_location = glGetUniformLocation(program, "rotate_yz");
	GLuint rotate3_location = glGetUniformLocation(program, "rotate_xy");

	add_random_metaballs(50 - metaballs.size());
	update_grid_indices();

	std::map<SDL_Keycode, bool> button_down;
	auto last_frame_start = std::chrono::high_resolution_clock::now();
    float time = 0.f;

	float xz_angle = 0.0f;
	float yz_angle = PI;
	float xy_angle = 0.0f;
	float scale = 0.5;
	float speed_of_rotation = 1.3;

	glEnable(GL_DEPTH_TEST);

	bool running = true;
	while (running) {
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
        time += dt;
		
		glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

		if (button_down[SDLK_LEFT]) {
			xz_angle -= speed_of_rotation * dt;
		} else if (button_down[SDLK_RIGHT]) {
			xz_angle += speed_of_rotation * dt;
		} else if (button_down[SDLK_UP]) {
			yz_angle += speed_of_rotation * dt;
		} else if (button_down[SDLK_DOWN]) {
			yz_angle -= speed_of_rotation * dt;
		} else if (button_down[SDLK_a]) {
			xy_angle += speed_of_rotation * dt;
		} else if (button_down[SDLK_d]) {
			xy_angle -= speed_of_rotation * dt;
		} else if (button_down[SDLK_w]) {
			number_of_isolines = std::min(number_of_isolines + 0.25f, 40.f);
			std::cout << "number_of_isolines = " << number_of_isolines << std::endl;
			update_isolines(min_z, max_z);
		} else if (button_down[SDLK_s]) {
			number_of_isolines = std::max(number_of_isolines - 0.25f, 0.f);
			std::cout << "number_of_isolines = " << number_of_isolines << std::endl;
			update_isolines(min_z, max_z);
		} else if (button_down[SDLK_k]) {
			grid_size = std::min(grid_size + 1, 150);
			std::cout << "grid_size = " << grid_size << std::endl;
			update_grid_indices();
		} else if (button_down[SDLK_l]) {
			grid_size = std::max(grid_size - 1, 4);
			std::cout << "grid_size = " << grid_size << std::endl;
			update_grid_indices();
		}

		float aspect_ratio_x = width * 1.0f / height;
		float aspect_ratio_y = 1.0f;
		// std::cout << "aspect_ratio_x = " << aspect_ratio_x << std::endl;
		if (aspect_ratio_x < 1.0f) {
			aspect_ratio_x = 1.0f;
			aspect_ratio_y = height * 1.0f / width;
		}
		
		float view[16] = {
            scale / aspect_ratio_x, 0.0, 0.0, 0.0,
            0.0, scale / aspect_ratio_y, 0.0, 0.0,
            0.0, 0.0, scale, 0.0,
            0.0, 0.0, 0.0, 1.0
        };

		float rotate_xz[16] = {
			cos(xz_angle), 0.f, -sin(xz_angle), 0.f,
			0.f, 1.f, 0.f, 0.0f,
			sin(xz_angle), 0.f, cos(xz_angle), 0.0f,
			0.f, 0.f, 0.f, 1.f,
		};

		float rotate_yz[16] = {
			1.f, 0.f, 0.f, 0.f,
			0.f, cos(yz_angle), -sin(yz_angle), 0.f,
			0.f, sin(yz_angle), cos(yz_angle), 0.f,
			0.f, 0.f, 0.f, 1.f,
		};

		float rotate_xy[16] = {
			cos(xy_angle), -sin(xy_angle), 0.f, 0.f,
			sin(xy_angle), cos(xy_angle), 0.f, 0.f,
			0.f, 0.f, 1.f, 0.f,
			0.f, 0.f, 0.f, 1.f,
		};

		update_metaballs(dt);

        glUseProgram(program);

		glBindVertexArray(VAO);
		// glBindBuffer(GL_ARRAY_BUFFER, VBO);
		glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, EBO);

		glUniformMatrix4fv(view_location, 1, GL_TRUE, view);
		glUniformMatrix4fv(rotate_location, 1, GL_TRUE, rotate_xz);
		glUniformMatrix4fv(rotate2_location, 1, GL_TRUE, rotate_yz);
		glUniformMatrix4fv(rotate3_location, 1, GL_TRUE, rotate_xy);

		glDrawElements(GL_TRIANGLES, grid_indices.size(), GL_UNSIGNED_INT, nullptr);

		glBindVertexArray(VAO2);
		// glBindBuffer(GL_ARRAY_BUFFER, VBO2);
		glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, EBO2);
		
		glDrawElements(GL_LINES, isoline_indices.size(), GL_UNSIGNED_INT, nullptr);

		SDL_GL_SwapWindow(window);
	}

	SDL_GL_DeleteContext(gl_context);
	SDL_DestroyWindow(window);
}
catch (std::exception const &e)
{
	std::cerr << e.what() << std::endl;
	return EXIT_FAILURE;
}
