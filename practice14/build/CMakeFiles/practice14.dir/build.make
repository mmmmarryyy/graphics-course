# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.27

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:

#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:

# Disable VCS-based implicit rules.
% : %,v

# Disable VCS-based implicit rules.
% : RCS/%

# Disable VCS-based implicit rules.
% : RCS/%,v

# Disable VCS-based implicit rules.
% : SCCS/s.%

# Disable VCS-based implicit rules.
% : s.%

.SUFFIXES: .hpux_make_needs_suffix_list

# Command-line flag to silence nested $(MAKE).
$(VERBOSE)MAKESILENT = -s

#Suppress display of executed commands.
$(VERBOSE).SILENT:

# A target that is always out of date.
cmake_force:
.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /opt/homebrew/Cellar/cmake/3.27.4/bin/cmake

# The command to remove a file.
RM = /opt/homebrew/Cellar/cmake/3.27.4/bin/cmake -E rm -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /Users/maria.barkovskaya/Documents/uni/comp_graph/graphics-course-practice/practice14

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /Users/maria.barkovskaya/Documents/uni/comp_graph/graphics-course-practice/practice14/build

# Include any dependencies generated for this target.
include CMakeFiles/practice14.dir/depend.make
# Include any dependencies generated by the compiler for this target.
include CMakeFiles/practice14.dir/compiler_depend.make

# Include the progress variables for this target.
include CMakeFiles/practice14.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/practice14.dir/flags.make

CMakeFiles/practice14.dir/main.cpp.o: CMakeFiles/practice14.dir/flags.make
CMakeFiles/practice14.dir/main.cpp.o: /Users/maria.barkovskaya/Documents/uni/comp_graph/graphics-course-practice/practice14/main.cpp
CMakeFiles/practice14.dir/main.cpp.o: CMakeFiles/practice14.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green --progress-dir=/Users/maria.barkovskaya/Documents/uni/comp_graph/graphics-course-practice/practice14/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object CMakeFiles/practice14.dir/main.cpp.o"
	/Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT CMakeFiles/practice14.dir/main.cpp.o -MF CMakeFiles/practice14.dir/main.cpp.o.d -o CMakeFiles/practice14.dir/main.cpp.o -c /Users/maria.barkovskaya/Documents/uni/comp_graph/graphics-course-practice/practice14/main.cpp

CMakeFiles/practice14.dir/main.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Preprocessing CXX source to CMakeFiles/practice14.dir/main.cpp.i"
	/Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /Users/maria.barkovskaya/Documents/uni/comp_graph/graphics-course-practice/practice14/main.cpp > CMakeFiles/practice14.dir/main.cpp.i

CMakeFiles/practice14.dir/main.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Compiling CXX source to assembly CMakeFiles/practice14.dir/main.cpp.s"
	/Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /Users/maria.barkovskaya/Documents/uni/comp_graph/graphics-course-practice/practice14/main.cpp -o CMakeFiles/practice14.dir/main.cpp.s

CMakeFiles/practice14.dir/gltf_loader.cpp.o: CMakeFiles/practice14.dir/flags.make
CMakeFiles/practice14.dir/gltf_loader.cpp.o: /Users/maria.barkovskaya/Documents/uni/comp_graph/graphics-course-practice/practice14/gltf_loader.cpp
CMakeFiles/practice14.dir/gltf_loader.cpp.o: CMakeFiles/practice14.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green --progress-dir=/Users/maria.barkovskaya/Documents/uni/comp_graph/graphics-course-practice/practice14/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Building CXX object CMakeFiles/practice14.dir/gltf_loader.cpp.o"
	/Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT CMakeFiles/practice14.dir/gltf_loader.cpp.o -MF CMakeFiles/practice14.dir/gltf_loader.cpp.o.d -o CMakeFiles/practice14.dir/gltf_loader.cpp.o -c /Users/maria.barkovskaya/Documents/uni/comp_graph/graphics-course-practice/practice14/gltf_loader.cpp

CMakeFiles/practice14.dir/gltf_loader.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Preprocessing CXX source to CMakeFiles/practice14.dir/gltf_loader.cpp.i"
	/Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /Users/maria.barkovskaya/Documents/uni/comp_graph/graphics-course-practice/practice14/gltf_loader.cpp > CMakeFiles/practice14.dir/gltf_loader.cpp.i

CMakeFiles/practice14.dir/gltf_loader.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Compiling CXX source to assembly CMakeFiles/practice14.dir/gltf_loader.cpp.s"
	/Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /Users/maria.barkovskaya/Documents/uni/comp_graph/graphics-course-practice/practice14/gltf_loader.cpp -o CMakeFiles/practice14.dir/gltf_loader.cpp.s

CMakeFiles/practice14.dir/stb_image.c.o: CMakeFiles/practice14.dir/flags.make
CMakeFiles/practice14.dir/stb_image.c.o: /Users/maria.barkovskaya/Documents/uni/comp_graph/graphics-course-practice/practice14/stb_image.c
CMakeFiles/practice14.dir/stb_image.c.o: CMakeFiles/practice14.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green --progress-dir=/Users/maria.barkovskaya/Documents/uni/comp_graph/graphics-course-practice/practice14/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_3) "Building C object CMakeFiles/practice14.dir/stb_image.c.o"
	/Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/bin/cc $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -MD -MT CMakeFiles/practice14.dir/stb_image.c.o -MF CMakeFiles/practice14.dir/stb_image.c.o.d -o CMakeFiles/practice14.dir/stb_image.c.o -c /Users/maria.barkovskaya/Documents/uni/comp_graph/graphics-course-practice/practice14/stb_image.c

CMakeFiles/practice14.dir/stb_image.c.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Preprocessing C source to CMakeFiles/practice14.dir/stb_image.c.i"
	/Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/bin/cc $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -E /Users/maria.barkovskaya/Documents/uni/comp_graph/graphics-course-practice/practice14/stb_image.c > CMakeFiles/practice14.dir/stb_image.c.i

CMakeFiles/practice14.dir/stb_image.c.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Compiling C source to assembly CMakeFiles/practice14.dir/stb_image.c.s"
	/Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/bin/cc $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -S /Users/maria.barkovskaya/Documents/uni/comp_graph/graphics-course-practice/practice14/stb_image.c -o CMakeFiles/practice14.dir/stb_image.c.s

CMakeFiles/practice14.dir/aabb.cpp.o: CMakeFiles/practice14.dir/flags.make
CMakeFiles/practice14.dir/aabb.cpp.o: /Users/maria.barkovskaya/Documents/uni/comp_graph/graphics-course-practice/practice14/aabb.cpp
CMakeFiles/practice14.dir/aabb.cpp.o: CMakeFiles/practice14.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green --progress-dir=/Users/maria.barkovskaya/Documents/uni/comp_graph/graphics-course-practice/practice14/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_4) "Building CXX object CMakeFiles/practice14.dir/aabb.cpp.o"
	/Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT CMakeFiles/practice14.dir/aabb.cpp.o -MF CMakeFiles/practice14.dir/aabb.cpp.o.d -o CMakeFiles/practice14.dir/aabb.cpp.o -c /Users/maria.barkovskaya/Documents/uni/comp_graph/graphics-course-practice/practice14/aabb.cpp

CMakeFiles/practice14.dir/aabb.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Preprocessing CXX source to CMakeFiles/practice14.dir/aabb.cpp.i"
	/Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /Users/maria.barkovskaya/Documents/uni/comp_graph/graphics-course-practice/practice14/aabb.cpp > CMakeFiles/practice14.dir/aabb.cpp.i

CMakeFiles/practice14.dir/aabb.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Compiling CXX source to assembly CMakeFiles/practice14.dir/aabb.cpp.s"
	/Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /Users/maria.barkovskaya/Documents/uni/comp_graph/graphics-course-practice/practice14/aabb.cpp -o CMakeFiles/practice14.dir/aabb.cpp.s

CMakeFiles/practice14.dir/frustum.cpp.o: CMakeFiles/practice14.dir/flags.make
CMakeFiles/practice14.dir/frustum.cpp.o: /Users/maria.barkovskaya/Documents/uni/comp_graph/graphics-course-practice/practice14/frustum.cpp
CMakeFiles/practice14.dir/frustum.cpp.o: CMakeFiles/practice14.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green --progress-dir=/Users/maria.barkovskaya/Documents/uni/comp_graph/graphics-course-practice/practice14/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_5) "Building CXX object CMakeFiles/practice14.dir/frustum.cpp.o"
	/Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT CMakeFiles/practice14.dir/frustum.cpp.o -MF CMakeFiles/practice14.dir/frustum.cpp.o.d -o CMakeFiles/practice14.dir/frustum.cpp.o -c /Users/maria.barkovskaya/Documents/uni/comp_graph/graphics-course-practice/practice14/frustum.cpp

CMakeFiles/practice14.dir/frustum.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Preprocessing CXX source to CMakeFiles/practice14.dir/frustum.cpp.i"
	/Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /Users/maria.barkovskaya/Documents/uni/comp_graph/graphics-course-practice/practice14/frustum.cpp > CMakeFiles/practice14.dir/frustum.cpp.i

CMakeFiles/practice14.dir/frustum.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Compiling CXX source to assembly CMakeFiles/practice14.dir/frustum.cpp.s"
	/Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /Users/maria.barkovskaya/Documents/uni/comp_graph/graphics-course-practice/practice14/frustum.cpp -o CMakeFiles/practice14.dir/frustum.cpp.s

# Object files for target practice14
practice14_OBJECTS = \
"CMakeFiles/practice14.dir/main.cpp.o" \
"CMakeFiles/practice14.dir/gltf_loader.cpp.o" \
"CMakeFiles/practice14.dir/stb_image.c.o" \
"CMakeFiles/practice14.dir/aabb.cpp.o" \
"CMakeFiles/practice14.dir/frustum.cpp.o"

# External object files for target practice14
practice14_EXTERNAL_OBJECTS =

practice14: CMakeFiles/practice14.dir/main.cpp.o
practice14: CMakeFiles/practice14.dir/gltf_loader.cpp.o
practice14: CMakeFiles/practice14.dir/stb_image.c.o
practice14: CMakeFiles/practice14.dir/aabb.cpp.o
practice14: CMakeFiles/practice14.dir/frustum.cpp.o
practice14: CMakeFiles/practice14.dir/build.make
practice14: /opt/homebrew/lib/libGLEW.2.2.0.dylib
practice14: /opt/homebrew/lib/libSDL2.dylib
practice14: CMakeFiles/practice14.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green --bold --progress-dir=/Users/maria.barkovskaya/Documents/uni/comp_graph/graphics-course-practice/practice14/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_6) "Linking CXX executable practice14"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/practice14.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/practice14.dir/build: practice14
.PHONY : CMakeFiles/practice14.dir/build

CMakeFiles/practice14.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/practice14.dir/cmake_clean.cmake
.PHONY : CMakeFiles/practice14.dir/clean

CMakeFiles/practice14.dir/depend:
	cd /Users/maria.barkovskaya/Documents/uni/comp_graph/graphics-course-practice/practice14/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /Users/maria.barkovskaya/Documents/uni/comp_graph/graphics-course-practice/practice14 /Users/maria.barkovskaya/Documents/uni/comp_graph/graphics-course-practice/practice14 /Users/maria.barkovskaya/Documents/uni/comp_graph/graphics-course-practice/practice14/build /Users/maria.barkovskaya/Documents/uni/comp_graph/graphics-course-practice/practice14/build /Users/maria.barkovskaya/Documents/uni/comp_graph/graphics-course-practice/practice14/build/CMakeFiles/practice14.dir/DependInfo.cmake "--color=$(COLOR)"
.PHONY : CMakeFiles/practice14.dir/depend

