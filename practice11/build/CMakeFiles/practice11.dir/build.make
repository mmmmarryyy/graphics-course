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
CMAKE_SOURCE_DIR = /Users/maria.barkovskaya/Documents/uni/comp_graph/graphics-course-practice/practice11

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /Users/maria.barkovskaya/Documents/uni/comp_graph/graphics-course-practice/practice11/build

# Include any dependencies generated for this target.
include CMakeFiles/practice11.dir/depend.make
# Include any dependencies generated by the compiler for this target.
include CMakeFiles/practice11.dir/compiler_depend.make

# Include the progress variables for this target.
include CMakeFiles/practice11.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/practice11.dir/flags.make

CMakeFiles/practice11.dir/main.cpp.o: CMakeFiles/practice11.dir/flags.make
CMakeFiles/practice11.dir/main.cpp.o: /Users/maria.barkovskaya/Documents/uni/comp_graph/graphics-course-practice/practice11/main.cpp
CMakeFiles/practice11.dir/main.cpp.o: CMakeFiles/practice11.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green --progress-dir=/Users/maria.barkovskaya/Documents/uni/comp_graph/graphics-course-practice/practice11/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object CMakeFiles/practice11.dir/main.cpp.o"
	/Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT CMakeFiles/practice11.dir/main.cpp.o -MF CMakeFiles/practice11.dir/main.cpp.o.d -o CMakeFiles/practice11.dir/main.cpp.o -c /Users/maria.barkovskaya/Documents/uni/comp_graph/graphics-course-practice/practice11/main.cpp

CMakeFiles/practice11.dir/main.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Preprocessing CXX source to CMakeFiles/practice11.dir/main.cpp.i"
	/Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /Users/maria.barkovskaya/Documents/uni/comp_graph/graphics-course-practice/practice11/main.cpp > CMakeFiles/practice11.dir/main.cpp.i

CMakeFiles/practice11.dir/main.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Compiling CXX source to assembly CMakeFiles/practice11.dir/main.cpp.s"
	/Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /Users/maria.barkovskaya/Documents/uni/comp_graph/graphics-course-practice/practice11/main.cpp -o CMakeFiles/practice11.dir/main.cpp.s

CMakeFiles/practice11.dir/obj_parser.cpp.o: CMakeFiles/practice11.dir/flags.make
CMakeFiles/practice11.dir/obj_parser.cpp.o: /Users/maria.barkovskaya/Documents/uni/comp_graph/graphics-course-practice/practice11/obj_parser.cpp
CMakeFiles/practice11.dir/obj_parser.cpp.o: CMakeFiles/practice11.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green --progress-dir=/Users/maria.barkovskaya/Documents/uni/comp_graph/graphics-course-practice/practice11/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Building CXX object CMakeFiles/practice11.dir/obj_parser.cpp.o"
	/Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT CMakeFiles/practice11.dir/obj_parser.cpp.o -MF CMakeFiles/practice11.dir/obj_parser.cpp.o.d -o CMakeFiles/practice11.dir/obj_parser.cpp.o -c /Users/maria.barkovskaya/Documents/uni/comp_graph/graphics-course-practice/practice11/obj_parser.cpp

CMakeFiles/practice11.dir/obj_parser.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Preprocessing CXX source to CMakeFiles/practice11.dir/obj_parser.cpp.i"
	/Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /Users/maria.barkovskaya/Documents/uni/comp_graph/graphics-course-practice/practice11/obj_parser.cpp > CMakeFiles/practice11.dir/obj_parser.cpp.i

CMakeFiles/practice11.dir/obj_parser.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Compiling CXX source to assembly CMakeFiles/practice11.dir/obj_parser.cpp.s"
	/Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /Users/maria.barkovskaya/Documents/uni/comp_graph/graphics-course-practice/practice11/obj_parser.cpp -o CMakeFiles/practice11.dir/obj_parser.cpp.s

CMakeFiles/practice11.dir/stb_image.c.o: CMakeFiles/practice11.dir/flags.make
CMakeFiles/practice11.dir/stb_image.c.o: /Users/maria.barkovskaya/Documents/uni/comp_graph/graphics-course-practice/practice11/stb_image.c
CMakeFiles/practice11.dir/stb_image.c.o: CMakeFiles/practice11.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green --progress-dir=/Users/maria.barkovskaya/Documents/uni/comp_graph/graphics-course-practice/practice11/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_3) "Building C object CMakeFiles/practice11.dir/stb_image.c.o"
	/Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/bin/cc $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -MD -MT CMakeFiles/practice11.dir/stb_image.c.o -MF CMakeFiles/practice11.dir/stb_image.c.o.d -o CMakeFiles/practice11.dir/stb_image.c.o -c /Users/maria.barkovskaya/Documents/uni/comp_graph/graphics-course-practice/practice11/stb_image.c

CMakeFiles/practice11.dir/stb_image.c.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Preprocessing C source to CMakeFiles/practice11.dir/stb_image.c.i"
	/Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/bin/cc $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -E /Users/maria.barkovskaya/Documents/uni/comp_graph/graphics-course-practice/practice11/stb_image.c > CMakeFiles/practice11.dir/stb_image.c.i

CMakeFiles/practice11.dir/stb_image.c.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Compiling C source to assembly CMakeFiles/practice11.dir/stb_image.c.s"
	/Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/bin/cc $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -S /Users/maria.barkovskaya/Documents/uni/comp_graph/graphics-course-practice/practice11/stb_image.c -o CMakeFiles/practice11.dir/stb_image.c.s

# Object files for target practice11
practice11_OBJECTS = \
"CMakeFiles/practice11.dir/main.cpp.o" \
"CMakeFiles/practice11.dir/obj_parser.cpp.o" \
"CMakeFiles/practice11.dir/stb_image.c.o"

# External object files for target practice11
practice11_EXTERNAL_OBJECTS =

practice11: CMakeFiles/practice11.dir/main.cpp.o
practice11: CMakeFiles/practice11.dir/obj_parser.cpp.o
practice11: CMakeFiles/practice11.dir/stb_image.c.o
practice11: CMakeFiles/practice11.dir/build.make
practice11: /opt/homebrew/lib/libGLEW.2.2.0.dylib
practice11: /opt/homebrew/lib/libSDL2.dylib
practice11: CMakeFiles/practice11.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green --bold --progress-dir=/Users/maria.barkovskaya/Documents/uni/comp_graph/graphics-course-practice/practice11/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_4) "Linking CXX executable practice11"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/practice11.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/practice11.dir/build: practice11
.PHONY : CMakeFiles/practice11.dir/build

CMakeFiles/practice11.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/practice11.dir/cmake_clean.cmake
.PHONY : CMakeFiles/practice11.dir/clean

CMakeFiles/practice11.dir/depend:
	cd /Users/maria.barkovskaya/Documents/uni/comp_graph/graphics-course-practice/practice11/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /Users/maria.barkovskaya/Documents/uni/comp_graph/graphics-course-practice/practice11 /Users/maria.barkovskaya/Documents/uni/comp_graph/graphics-course-practice/practice11 /Users/maria.barkovskaya/Documents/uni/comp_graph/graphics-course-practice/practice11/build /Users/maria.barkovskaya/Documents/uni/comp_graph/graphics-course-practice/practice11/build /Users/maria.barkovskaya/Documents/uni/comp_graph/graphics-course-practice/practice11/build/CMakeFiles/practice11.dir/DependInfo.cmake "--color=$(COLOR)"
.PHONY : CMakeFiles/practice11.dir/depend

