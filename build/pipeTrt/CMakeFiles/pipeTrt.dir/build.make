# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.10

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:


#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:


# Remove some rules from gmake that .SUFFIXES does not remove.
SUFFIXES =

.SUFFIXES: .hpux_make_needs_suffix_list


# Suppress display of executed commands.
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
RM = /usr/bin/cmake -E remove -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /home/nvidia/PaulPeters/tensorrtPipe

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/nvidia/PaulPeters/tensorrtPipe/build

# Include any dependencies generated for this target.
include pipeTrt/CMakeFiles/pipeTrt.dir/depend.make

# Include the progress variables for this target.
include pipeTrt/CMakeFiles/pipeTrt.dir/progress.make

# Include the compile flags for this target's objects.
include pipeTrt/CMakeFiles/pipeTrt.dir/flags.make

pipeTrt/CMakeFiles/pipeTrt.dir/src/PipeTrt.cpp.o: pipeTrt/CMakeFiles/pipeTrt.dir/flags.make
pipeTrt/CMakeFiles/pipeTrt.dir/src/PipeTrt.cpp.o: ../pipeTrt/src/PipeTrt.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/nvidia/PaulPeters/tensorrtPipe/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object pipeTrt/CMakeFiles/pipeTrt.dir/src/PipeTrt.cpp.o"
	cd /home/nvidia/PaulPeters/tensorrtPipe/build/pipeTrt && /usr/bin/g++-7  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/pipeTrt.dir/src/PipeTrt.cpp.o -c /home/nvidia/PaulPeters/tensorrtPipe/pipeTrt/src/PipeTrt.cpp

pipeTrt/CMakeFiles/pipeTrt.dir/src/PipeTrt.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/pipeTrt.dir/src/PipeTrt.cpp.i"
	$(CMAKE_COMMAND) -E cmake_unimplemented_variable CMAKE_CXX_CREATE_PREPROCESSED_SOURCE

pipeTrt/CMakeFiles/pipeTrt.dir/src/PipeTrt.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/pipeTrt.dir/src/PipeTrt.cpp.s"
	$(CMAKE_COMMAND) -E cmake_unimplemented_variable CMAKE_CXX_CREATE_ASSEMBLY_SOURCE

pipeTrt/CMakeFiles/pipeTrt.dir/src/PipeTrt.cpp.o.requires:

.PHONY : pipeTrt/CMakeFiles/pipeTrt.dir/src/PipeTrt.cpp.o.requires

pipeTrt/CMakeFiles/pipeTrt.dir/src/PipeTrt.cpp.o.provides: pipeTrt/CMakeFiles/pipeTrt.dir/src/PipeTrt.cpp.o.requires
	$(MAKE) -f pipeTrt/CMakeFiles/pipeTrt.dir/build.make pipeTrt/CMakeFiles/pipeTrt.dir/src/PipeTrt.cpp.o.provides.build
.PHONY : pipeTrt/CMakeFiles/pipeTrt.dir/src/PipeTrt.cpp.o.provides

pipeTrt/CMakeFiles/pipeTrt.dir/src/PipeTrt.cpp.o.provides.build: pipeTrt/CMakeFiles/pipeTrt.dir/src/PipeTrt.cpp.o


# Object files for target pipeTrt
pipeTrt_OBJECTS = \
"CMakeFiles/pipeTrt.dir/src/PipeTrt.cpp.o"

# External object files for target pipeTrt
pipeTrt_EXTERNAL_OBJECTS =

../lib/libpipeTrt.a: pipeTrt/CMakeFiles/pipeTrt.dir/src/PipeTrt.cpp.o
../lib/libpipeTrt.a: pipeTrt/CMakeFiles/pipeTrt.dir/build.make
../lib/libpipeTrt.a: pipeTrt/CMakeFiles/pipeTrt.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/nvidia/PaulPeters/tensorrtPipe/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX static library ../../lib/libpipeTrt.a"
	cd /home/nvidia/PaulPeters/tensorrtPipe/build/pipeTrt && $(CMAKE_COMMAND) -P CMakeFiles/pipeTrt.dir/cmake_clean_target.cmake
	cd /home/nvidia/PaulPeters/tensorrtPipe/build/pipeTrt && $(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/pipeTrt.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
pipeTrt/CMakeFiles/pipeTrt.dir/build: ../lib/libpipeTrt.a

.PHONY : pipeTrt/CMakeFiles/pipeTrt.dir/build

pipeTrt/CMakeFiles/pipeTrt.dir/requires: pipeTrt/CMakeFiles/pipeTrt.dir/src/PipeTrt.cpp.o.requires

.PHONY : pipeTrt/CMakeFiles/pipeTrt.dir/requires

pipeTrt/CMakeFiles/pipeTrt.dir/clean:
	cd /home/nvidia/PaulPeters/tensorrtPipe/build/pipeTrt && $(CMAKE_COMMAND) -P CMakeFiles/pipeTrt.dir/cmake_clean.cmake
.PHONY : pipeTrt/CMakeFiles/pipeTrt.dir/clean

pipeTrt/CMakeFiles/pipeTrt.dir/depend:
	cd /home/nvidia/PaulPeters/tensorrtPipe/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/nvidia/PaulPeters/tensorrtPipe /home/nvidia/PaulPeters/tensorrtPipe/pipeTrt /home/nvidia/PaulPeters/tensorrtPipe/build /home/nvidia/PaulPeters/tensorrtPipe/build/pipeTrt /home/nvidia/PaulPeters/tensorrtPipe/build/pipeTrt/CMakeFiles/pipeTrt.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : pipeTrt/CMakeFiles/pipeTrt.dir/depend

