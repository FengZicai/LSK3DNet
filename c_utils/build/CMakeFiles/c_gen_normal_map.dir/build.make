# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.19

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
CMAKE_COMMAND = /usr/bin/cmake

# The command to remove a file.
RM = /usr/bin/cmake -E rm -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /home/tuofeng/PVKD/c_utils

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/tuofeng/PVKD/c_utils/build

# Include any dependencies generated for this target.
include CMakeFiles/c_gen_normal_map.dir/depend.make

# Include the progress variables for this target.
include CMakeFiles/c_gen_normal_map.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/c_gen_normal_map.dir/flags.make

CMakeFiles/c_gen_normal_map.dir/src/c_gen_normal_map.cpp.o: CMakeFiles/c_gen_normal_map.dir/flags.make
CMakeFiles/c_gen_normal_map.dir/src/c_gen_normal_map.cpp.o: ../src/c_gen_normal_map.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/tuofeng/PVKD/c_utils/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object CMakeFiles/c_gen_normal_map.dir/src/c_gen_normal_map.cpp.o"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/c_gen_normal_map.dir/src/c_gen_normal_map.cpp.o -c /home/tuofeng/PVKD/c_utils/src/c_gen_normal_map.cpp

CMakeFiles/c_gen_normal_map.dir/src/c_gen_normal_map.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/c_gen_normal_map.dir/src/c_gen_normal_map.cpp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/tuofeng/PVKD/c_utils/src/c_gen_normal_map.cpp > CMakeFiles/c_gen_normal_map.dir/src/c_gen_normal_map.cpp.i

CMakeFiles/c_gen_normal_map.dir/src/c_gen_normal_map.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/c_gen_normal_map.dir/src/c_gen_normal_map.cpp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/tuofeng/PVKD/c_utils/src/c_gen_normal_map.cpp -o CMakeFiles/c_gen_normal_map.dir/src/c_gen_normal_map.cpp.s

# Object files for target c_gen_normal_map
c_gen_normal_map_OBJECTS = \
"CMakeFiles/c_gen_normal_map.dir/src/c_gen_normal_map.cpp.o"

# External object files for target c_gen_normal_map
c_gen_normal_map_EXTERNAL_OBJECTS =

c_gen_normal_map.cpython-37m-x86_64-linux-gnu.so: CMakeFiles/c_gen_normal_map.dir/src/c_gen_normal_map.cpp.o
c_gen_normal_map.cpython-37m-x86_64-linux-gnu.so: CMakeFiles/c_gen_normal_map.dir/build.make
c_gen_normal_map.cpython-37m-x86_64-linux-gnu.so: CMakeFiles/c_gen_normal_map.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/tuofeng/PVKD/c_utils/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX shared module c_gen_normal_map.cpython-37m-x86_64-linux-gnu.so"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/c_gen_normal_map.dir/link.txt --verbose=$(VERBOSE)
	/usr/bin/strip /home/tuofeng/PVKD/c_utils/build/c_gen_normal_map.cpython-37m-x86_64-linux-gnu.so

# Rule to build all files generated by this target.
CMakeFiles/c_gen_normal_map.dir/build: c_gen_normal_map.cpython-37m-x86_64-linux-gnu.so

.PHONY : CMakeFiles/c_gen_normal_map.dir/build

CMakeFiles/c_gen_normal_map.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/c_gen_normal_map.dir/cmake_clean.cmake
.PHONY : CMakeFiles/c_gen_normal_map.dir/clean

CMakeFiles/c_gen_normal_map.dir/depend:
	cd /home/tuofeng/PVKD/c_utils/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/tuofeng/PVKD/c_utils /home/tuofeng/PVKD/c_utils /home/tuofeng/PVKD/c_utils/build /home/tuofeng/PVKD/c_utils/build /home/tuofeng/PVKD/c_utils/build/CMakeFiles/c_gen_normal_map.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/c_gen_normal_map.dir/depend

