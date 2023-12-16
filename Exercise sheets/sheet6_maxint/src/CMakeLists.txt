# Single
add_executable(reduce-single main.cu reduce-single.cu)

# Multi
add_executable(reduce-multi main.cu reduce-multi.cu)

# MPI
find_package(MPI REQUIRED)
add_executable(reduce-mpi main.cu reduce-mpi.cu)
target_link_libraries(reduce-mpi MPI::MPI_C)
target_compile_options(reduce-mpi PRIVATE -DUSE_MPI=1)
