# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.6

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
CMAKE_COMMAND = /Applications/CLion.app/Contents/bin/cmake/bin/cmake

# The command to remove a file.
RM = /Applications/CLion.app/Contents/bin/cmake/bin/cmake -E remove -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /Users/sunxiaofei/ClionProjects/dynet-language-model

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /Users/sunxiaofei/ClionProjects/dynet-language-model/cmake-build-debug

# Include any dependencies generated for this target.
include CMakeFiles/dlne_main.dir/depend.make

# Include the progress variables for this target.
include CMakeFiles/dlne_main.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/dlne_main.dir/flags.make

CMakeFiles/dlne_main.dir/src/train_rnnlm-mp.cc.o: CMakeFiles/dlne_main.dir/flags.make
CMakeFiles/dlne_main.dir/src/train_rnnlm-mp.cc.o: ../src/train_rnnlm-mp.cc
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/Users/sunxiaofei/ClionProjects/dynet-language-model/cmake-build-debug/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object CMakeFiles/dlne_main.dir/src/train_rnnlm-mp.cc.o"
	/Library/Developer/CommandLineTools/usr/bin/c++   $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/dlne_main.dir/src/train_rnnlm-mp.cc.o -c /Users/sunxiaofei/ClionProjects/dynet-language-model/src/train_rnnlm-mp.cc

CMakeFiles/dlne_main.dir/src/train_rnnlm-mp.cc.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/dlne_main.dir/src/train_rnnlm-mp.cc.i"
	/Library/Developer/CommandLineTools/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /Users/sunxiaofei/ClionProjects/dynet-language-model/src/train_rnnlm-mp.cc > CMakeFiles/dlne_main.dir/src/train_rnnlm-mp.cc.i

CMakeFiles/dlne_main.dir/src/train_rnnlm-mp.cc.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/dlne_main.dir/src/train_rnnlm-mp.cc.s"
	/Library/Developer/CommandLineTools/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /Users/sunxiaofei/ClionProjects/dynet-language-model/src/train_rnnlm-mp.cc -o CMakeFiles/dlne_main.dir/src/train_rnnlm-mp.cc.s

CMakeFiles/dlne_main.dir/src/train_rnnlm-mp.cc.o.requires:

.PHONY : CMakeFiles/dlne_main.dir/src/train_rnnlm-mp.cc.o.requires

CMakeFiles/dlne_main.dir/src/train_rnnlm-mp.cc.o.provides: CMakeFiles/dlne_main.dir/src/train_rnnlm-mp.cc.o.requires
	$(MAKE) -f CMakeFiles/dlne_main.dir/build.make CMakeFiles/dlne_main.dir/src/train_rnnlm-mp.cc.o.provides.build
.PHONY : CMakeFiles/dlne_main.dir/src/train_rnnlm-mp.cc.o.provides

CMakeFiles/dlne_main.dir/src/train_rnnlm-mp.cc.o.provides.build: CMakeFiles/dlne_main.dir/src/train_rnnlm-mp.cc.o


# Object files for target dlne_main
dlne_main_OBJECTS = \
"CMakeFiles/dlne_main.dir/src/train_rnnlm-mp.cc.o"

# External object files for target dlne_main
dlne_main_EXTERNAL_OBJECTS =

dlne_main: CMakeFiles/dlne_main.dir/src/train_rnnlm-mp.cc.o
dlne_main: CMakeFiles/dlne_main.dir/build.make
dlne_main: /usr/local/Cellar/boost/1.62.0/lib/libboost_program_options-mt.dylib
dlne_main: /usr/local/Cellar/boost/1.62.0/lib/libboost_serialization-mt.dylib
dlne_main: CMakeFiles/dlne_main.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/Users/sunxiaofei/ClionProjects/dynet-language-model/cmake-build-debug/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX executable dlne_main"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/dlne_main.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/dlne_main.dir/build: dlne_main

.PHONY : CMakeFiles/dlne_main.dir/build

CMakeFiles/dlne_main.dir/requires: CMakeFiles/dlne_main.dir/src/train_rnnlm-mp.cc.o.requires

.PHONY : CMakeFiles/dlne_main.dir/requires

CMakeFiles/dlne_main.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/dlne_main.dir/cmake_clean.cmake
.PHONY : CMakeFiles/dlne_main.dir/clean

CMakeFiles/dlne_main.dir/depend:
	cd /Users/sunxiaofei/ClionProjects/dynet-language-model/cmake-build-debug && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /Users/sunxiaofei/ClionProjects/dynet-language-model /Users/sunxiaofei/ClionProjects/dynet-language-model /Users/sunxiaofei/ClionProjects/dynet-language-model/cmake-build-debug /Users/sunxiaofei/ClionProjects/dynet-language-model/cmake-build-debug /Users/sunxiaofei/ClionProjects/dynet-language-model/cmake-build-debug/CMakeFiles/dlne_main.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/dlne_main.dir/depend
