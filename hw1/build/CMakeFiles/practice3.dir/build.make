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
CMAKE_SOURCE_DIR = /Users/maria.barkovskaya/Documents/uni/comp_graph/graphics-course-practice/hw1

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /Users/maria.barkovskaya/Documents/uni/comp_graph/graphics-course-practice/hw1/build

# Include any dependencies generated for this target.
include CMakeFiles/practice3.dir/depend.make
# Include any dependencies generated by the compiler for this target.
include CMakeFiles/practice3.dir/compiler_depend.make

# Include the progress variables for this target.
include CMakeFiles/practice3.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/practice3.dir/flags.make

CMakeFiles/practice3.dir/main.cpp.o: CMakeFiles/practice3.dir/flags.make
CMakeFiles/practice3.dir/main.cpp.o: /Users/maria.barkovskaya/Documents/uni/comp_graph/graphics-course-practice/hw1/main.cpp
CMakeFiles/practice3.dir/main.cpp.o: CMakeFiles/practice3.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green --progress-dir=/Users/maria.barkovskaya/Documents/uni/comp_graph/graphics-course-practice/hw1/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object CMakeFiles/practice3.dir/main.cpp.o"
	/Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT CMakeFiles/practice3.dir/main.cpp.o -MF CMakeFiles/practice3.dir/main.cpp.o.d -o CMakeFiles/practice3.dir/main.cpp.o -c /Users/maria.barkovskaya/Documents/uni/comp_graph/graphics-course-practice/hw1/main.cpp

CMakeFiles/practice3.dir/main.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Preprocessing CXX source to CMakeFiles/practice3.dir/main.cpp.i"
	/Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /Users/maria.barkovskaya/Documents/uni/comp_graph/graphics-course-practice/hw1/main.cpp > CMakeFiles/practice3.dir/main.cpp.i

CMakeFiles/practice3.dir/main.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Compiling CXX source to assembly CMakeFiles/practice3.dir/main.cpp.s"
	/Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /Users/maria.barkovskaya/Documents/uni/comp_graph/graphics-course-practice/hw1/main.cpp -o CMakeFiles/practice3.dir/main.cpp.s

# Object files for target practice3
practice3_OBJECTS = \
"CMakeFiles/practice3.dir/main.cpp.o"

# External object files for target practice3
practice3_EXTERNAL_OBJECTS =

practice3: CMakeFiles/practice3.dir/main.cpp.o
practice3: CMakeFiles/practice3.dir/build.make
practice3: /opt/homebrew/lib/libGLEW.2.2.0.dylib
practice3: /opt/homebrew/lib/libSDL2.dylib
practice3: CMakeFiles/practice3.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green --bold --progress-dir=/Users/maria.barkovskaya/Documents/uni/comp_graph/graphics-course-practice/hw1/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX executable practice3"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/practice3.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/practice3.dir/build: practice3
.PHONY : CMakeFiles/practice3.dir/build

CMakeFiles/practice3.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/practice3.dir/cmake_clean.cmake
.PHONY : CMakeFiles/practice3.dir/clean

CMakeFiles/practice3.dir/depend:
	cd /Users/maria.barkovskaya/Documents/uni/comp_graph/graphics-course-practice/hw1/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /Users/maria.barkovskaya/Documents/uni/comp_graph/graphics-course-practice/hw1 /Users/maria.barkovskaya/Documents/uni/comp_graph/graphics-course-practice/hw1 /Users/maria.barkovskaya/Documents/uni/comp_graph/graphics-course-practice/hw1/build /Users/maria.barkovskaya/Documents/uni/comp_graph/graphics-course-practice/hw1/build /Users/maria.barkovskaya/Documents/uni/comp_graph/graphics-course-practice/hw1/build/CMakeFiles/practice3.dir/DependInfo.cmake "--color=$(COLOR)"
.PHONY : CMakeFiles/practice3.dir/depend

