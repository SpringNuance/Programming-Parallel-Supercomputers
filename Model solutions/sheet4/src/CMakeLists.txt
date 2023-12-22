#Run files
enable_language(CUDA)
add_executable(quicksort-serial main.cu quicksort.cu)
add_executable(quicksort-gpu main.cu main.cu quicksort_gpu.cu)
add_executable(quicksort-distributed-bcast main.cu quicksort_distributed_bcast.cu)
add_executable(quicksort-distributed-rdma main.cu quicksort_distributed_rdma.cu)
add_executable(quicksort-distributed-gpu main.cu quicksort_distributed_gpu.cu)

# MPI
find_package(MPI REQUIRED)

target_link_libraries(quicksort-distributed-bcast MPI::MPI_C)
target_link_libraries(quicksort-distributed-rdma MPI::MPI_C)
target_link_libraries(quicksort-distributed-gpu MPI::MPI_C)

target_compile_options(quicksort-serial PRIVATE -DGPU=0 -DDISTRIBUTED=0)
target_compile_options(quicksort-gpu PRIVATE -DGPU=1 -DDISTRIBUTED=0)
target_compile_options(quicksort-distributed-rdma PRIVATE -DGPU=0 -DDISTRIBUTED=1)
target_compile_options(quicksort-distributed-bcast PRIVATE -DGPU=0 -DDISTRIBUTED=1)
target_compile_options(quicksort-distributed-gpu PRIVATE -DGPU=1 -DDISTRIBUTED=1)

