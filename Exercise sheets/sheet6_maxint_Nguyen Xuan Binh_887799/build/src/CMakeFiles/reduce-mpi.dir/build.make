# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.26

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
CMAKE_COMMAND = /share/apps/scibuilder-spack/aalto-centos7/2023-01/software/linux-centos7-haswell/gcc-11.3.0/cmake-3.26.3-hvnw6he/bin/cmake

# The command to remove a file.
RM = /share/apps/scibuilder-spack/aalto-centos7/2023-01/software/linux-centos7-haswell/gcc-11.3.0/cmake-3.26.3-hvnw6he/bin/cmake -E rm -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /scratch/courses/programming_parallel_supercomputers_2023/nguyenb5/sheet6

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /scratch/courses/programming_parallel_supercomputers_2023/nguyenb5/sheet6/build

# Include any dependencies generated for this target.
include src/CMakeFiles/reduce-mpi.dir/depend.make
# Include any dependencies generated by the compiler for this target.
include src/CMakeFiles/reduce-mpi.dir/compiler_depend.make

# Include the progress variables for this target.
include src/CMakeFiles/reduce-mpi.dir/progress.make

# Include the compile flags for this target's objects.
include src/CMakeFiles/reduce-mpi.dir/flags.make

src/CMakeFiles/reduce-mpi.dir/main.cu.o: src/CMakeFiles/reduce-mpi.dir/flags.make
src/CMakeFiles/reduce-mpi.dir/main.cu.o: /scratch/courses/programming_parallel_supercomputers_2023/nguyenb5/sheet6/src/main.cu
src/CMakeFiles/reduce-mpi.dir/main.cu.o: src/CMakeFiles/reduce-mpi.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/scratch/courses/programming_parallel_supercomputers_2023/nguyenb5/sheet6/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CUDA object src/CMakeFiles/reduce-mpi.dir/main.cu.o"
	cd /scratch/courses/programming_parallel_supercomputers_2023/nguyenb5/sheet6/build/src && /share/apps/scibuilder-spack/aalto-centos7/2023-01/software/linux-centos7-haswell/gcc-11.3.0/cuda-11.8.0-gff3eyf/bin/nvcc -forward-unknown-to-host-compiler $(CUDA_DEFINES) $(CUDA_INCLUDES) $(CUDA_FLAGS) -MD -MT src/CMakeFiles/reduce-mpi.dir/main.cu.o -MF CMakeFiles/reduce-mpi.dir/main.cu.o.d -x cu -c /scratch/courses/programming_parallel_supercomputers_2023/nguyenb5/sheet6/src/main.cu -o CMakeFiles/reduce-mpi.dir/main.cu.o

src/CMakeFiles/reduce-mpi.dir/main.cu.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CUDA source to CMakeFiles/reduce-mpi.dir/main.cu.i"
	$(CMAKE_COMMAND) -E cmake_unimplemented_variable CMAKE_CUDA_CREATE_PREPROCESSED_SOURCE

src/CMakeFiles/reduce-mpi.dir/main.cu.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CUDA source to assembly CMakeFiles/reduce-mpi.dir/main.cu.s"
	$(CMAKE_COMMAND) -E cmake_unimplemented_variable CMAKE_CUDA_CREATE_ASSEMBLY_SOURCE

src/CMakeFiles/reduce-mpi.dir/reduce-mpi.cu.o: src/CMakeFiles/reduce-mpi.dir/flags.make
src/CMakeFiles/reduce-mpi.dir/reduce-mpi.cu.o: /scratch/courses/programming_parallel_supercomputers_2023/nguyenb5/sheet6/src/reduce-mpi.cu
src/CMakeFiles/reduce-mpi.dir/reduce-mpi.cu.o: src/CMakeFiles/reduce-mpi.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/scratch/courses/programming_parallel_supercomputers_2023/nguyenb5/sheet6/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Building CUDA object src/CMakeFiles/reduce-mpi.dir/reduce-mpi.cu.o"
	cd /scratch/courses/programming_parallel_supercomputers_2023/nguyenb5/sheet6/build/src && /share/apps/scibuilder-spack/aalto-centos7/2023-01/software/linux-centos7-haswell/gcc-11.3.0/cuda-11.8.0-gff3eyf/bin/nvcc -forward-unknown-to-host-compiler $(CUDA_DEFINES) $(CUDA_INCLUDES) $(CUDA_FLAGS) -MD -MT src/CMakeFiles/reduce-mpi.dir/reduce-mpi.cu.o -MF CMakeFiles/reduce-mpi.dir/reduce-mpi.cu.o.d -x cu -c /scratch/courses/programming_parallel_supercomputers_2023/nguyenb5/sheet6/src/reduce-mpi.cu -o CMakeFiles/reduce-mpi.dir/reduce-mpi.cu.o

src/CMakeFiles/reduce-mpi.dir/reduce-mpi.cu.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CUDA source to CMakeFiles/reduce-mpi.dir/reduce-mpi.cu.i"
	$(CMAKE_COMMAND) -E cmake_unimplemented_variable CMAKE_CUDA_CREATE_PREPROCESSED_SOURCE

src/CMakeFiles/reduce-mpi.dir/reduce-mpi.cu.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CUDA source to assembly CMakeFiles/reduce-mpi.dir/reduce-mpi.cu.s"
	$(CMAKE_COMMAND) -E cmake_unimplemented_variable CMAKE_CUDA_CREATE_ASSEMBLY_SOURCE

# Object files for target reduce-mpi
reduce__mpi_OBJECTS = \
"CMakeFiles/reduce-mpi.dir/main.cu.o" \
"CMakeFiles/reduce-mpi.dir/reduce-mpi.cu.o"

# External object files for target reduce-mpi
reduce__mpi_EXTERNAL_OBJECTS =

reduce-mpi: src/CMakeFiles/reduce-mpi.dir/main.cu.o
reduce-mpi: src/CMakeFiles/reduce-mpi.dir/reduce-mpi.cu.o
reduce-mpi: src/CMakeFiles/reduce-mpi.dir/build.make
reduce-mpi: /share/apps/scibuilder-spack/aalto-centos7/2023-01/software/linux-centos7-haswell/gcc-11.3.0/openmpi-4.1.5-3qsfx7s/lib/libmpi.so
reduce-mpi: src/CMakeFiles/reduce-mpi.dir/linkLibs.rsp
reduce-mpi: src/CMakeFiles/reduce-mpi.dir/objects1.rsp
reduce-mpi: src/CMakeFiles/reduce-mpi.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/scratch/courses/programming_parallel_supercomputers_2023/nguyenb5/sheet6/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_3) "Linking CUDA executable ../reduce-mpi"
	cd /scratch/courses/programming_parallel_supercomputers_2023/nguyenb5/sheet6/build/src && $(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/reduce-mpi.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
src/CMakeFiles/reduce-mpi.dir/build: reduce-mpi
.PHONY : src/CMakeFiles/reduce-mpi.dir/build

src/CMakeFiles/reduce-mpi.dir/clean:
	cd /scratch/courses/programming_parallel_supercomputers_2023/nguyenb5/sheet6/build/src && $(CMAKE_COMMAND) -P CMakeFiles/reduce-mpi.dir/cmake_clean.cmake
.PHONY : src/CMakeFiles/reduce-mpi.dir/clean

src/CMakeFiles/reduce-mpi.dir/depend:
	cd /scratch/courses/programming_parallel_supercomputers_2023/nguyenb5/sheet6/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /scratch/courses/programming_parallel_supercomputers_2023/nguyenb5/sheet6 /scratch/courses/programming_parallel_supercomputers_2023/nguyenb5/sheet6/src /scratch/courses/programming_parallel_supercomputers_2023/nguyenb5/sheet6/build /scratch/courses/programming_parallel_supercomputers_2023/nguyenb5/sheet6/build/src /scratch/courses/programming_parallel_supercomputers_2023/nguyenb5/sheet6/build/src/CMakeFiles/reduce-mpi.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : src/CMakeFiles/reduce-mpi.dir/depend

