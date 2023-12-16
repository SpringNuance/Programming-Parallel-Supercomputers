set(CMAKE_CUDA_COMPILER "/share/apps/scibuilder-spack/aalto-centos7/2023-01/software/linux-centos7-haswell/gcc-11.3.0/cuda-11.8.0-gff3eyf/bin/nvcc")
set(CMAKE_CUDA_HOST_COMPILER "")
set(CMAKE_CUDA_HOST_LINK_LAUNCHER "/share/apps/scibuilder-spack/aalto-centos7/2023-01-compilers/software/linux-centos7-haswell/gcc-4.8.5/gcc-11.3.0-7prhbnn/bin/g++")
set(CMAKE_CUDA_COMPILER_ID "NVIDIA")
set(CMAKE_CUDA_COMPILER_VERSION "11.8.89")
set(CMAKE_CUDA_DEVICE_LINKER "/share/apps/scibuilder-spack/aalto-centos7/2023-01/software/linux-centos7-haswell/gcc-11.3.0/cuda-11.8.0-gff3eyf/bin/nvlink")
set(CMAKE_CUDA_FATBINARY "/share/apps/scibuilder-spack/aalto-centos7/2023-01/software/linux-centos7-haswell/gcc-11.3.0/cuda-11.8.0-gff3eyf/bin/fatbinary")
set(CMAKE_CUDA_STANDARD_COMPUTED_DEFAULT "17")
set(CMAKE_CUDA_EXTENSIONS_COMPUTED_DEFAULT "ON")
set(CMAKE_CUDA_COMPILE_FEATURES "cuda_std_03;cuda_std_11;cuda_std_14;cuda_std_17")
set(CMAKE_CUDA03_COMPILE_FEATURES "cuda_std_03")
set(CMAKE_CUDA11_COMPILE_FEATURES "cuda_std_11")
set(CMAKE_CUDA14_COMPILE_FEATURES "cuda_std_14")
set(CMAKE_CUDA17_COMPILE_FEATURES "cuda_std_17")
set(CMAKE_CUDA20_COMPILE_FEATURES "")
set(CMAKE_CUDA23_COMPILE_FEATURES "")

set(CMAKE_CUDA_PLATFORM_ID "Linux")
set(CMAKE_CUDA_SIMULATE_ID "GNU")
set(CMAKE_CUDA_COMPILER_FRONTEND_VARIANT "")
set(CMAKE_CUDA_SIMULATE_VERSION "11.3")



set(CMAKE_CUDA_COMPILER_ENV_VAR "CUDACXX")
set(CMAKE_CUDA_HOST_COMPILER_ENV_VAR "CUDAHOSTCXX")

set(CMAKE_CUDA_COMPILER_LOADED 1)
set(CMAKE_CUDA_COMPILER_ID_RUN 1)
set(CMAKE_CUDA_SOURCE_FILE_EXTENSIONS cu)
set(CMAKE_CUDA_LINKER_PREFERENCE 15)
set(CMAKE_CUDA_LINKER_PREFERENCE_PROPAGATES 1)

set(CMAKE_CUDA_SIZEOF_DATA_PTR "8")
set(CMAKE_CUDA_COMPILER_ABI "ELF")
set(CMAKE_CUDA_BYTE_ORDER "LITTLE_ENDIAN")
set(CMAKE_CUDA_LIBRARY_ARCHITECTURE "")

if(CMAKE_CUDA_SIZEOF_DATA_PTR)
  set(CMAKE_SIZEOF_VOID_P "${CMAKE_CUDA_SIZEOF_DATA_PTR}")
endif()

if(CMAKE_CUDA_COMPILER_ABI)
  set(CMAKE_INTERNAL_PLATFORM_ABI "${CMAKE_CUDA_COMPILER_ABI}")
endif()

if(CMAKE_CUDA_LIBRARY_ARCHITECTURE)
  set(CMAKE_LIBRARY_ARCHITECTURE "")
endif()

set(CMAKE_CUDA_COMPILER_TOOLKIT_ROOT "/share/apps/scibuilder-spack/aalto-centos7/2023-01/software/linux-centos7-haswell/gcc-11.3.0/cuda-11.8.0-gff3eyf")
set(CMAKE_CUDA_COMPILER_TOOLKIT_LIBRARY_ROOT "/share/apps/scibuilder-spack/aalto-centos7/2023-01/software/linux-centos7-haswell/gcc-11.3.0/cuda-11.8.0-gff3eyf")
set(CMAKE_CUDA_COMPILER_TOOLKIT_VERSION "11.8.89")
set(CMAKE_CUDA_COMPILER_LIBRARY_ROOT "/share/apps/scibuilder-spack/aalto-centos7/2023-01/software/linux-centos7-haswell/gcc-11.3.0/cuda-11.8.0-gff3eyf")

set(CMAKE_CUDA_ARCHITECTURES_ALL "35-real;37-real;50-real;52-real;53-real;60-real;61-real;62-real;70-real;72-real;75-real;80-real;86-real;87-real;89-real;90")
set(CMAKE_CUDA_ARCHITECTURES_ALL_MAJOR "35-real;50-real;60-real;70-real;80-real;90")
set(CMAKE_CUDA_ARCHITECTURES_NATIVE "")

set(CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES "/share/apps/scibuilder-spack/aalto-centos7/2023-01/software/linux-centos7-haswell/gcc-11.3.0/cuda-11.8.0-gff3eyf/targets/x86_64-linux/include")

set(CMAKE_CUDA_HOST_IMPLICIT_LINK_LIBRARIES "")
set(CMAKE_CUDA_HOST_IMPLICIT_LINK_DIRECTORIES "/share/apps/scibuilder-spack/aalto-centos7/2023-01/software/linux-centos7-haswell/gcc-11.3.0/cuda-11.8.0-gff3eyf/targets/x86_64-linux/lib/stubs;/share/apps/scibuilder-spack/aalto-centos7/2023-01/software/linux-centos7-haswell/gcc-11.3.0/cuda-11.8.0-gff3eyf/targets/x86_64-linux/lib")
set(CMAKE_CUDA_HOST_IMPLICIT_LINK_FRAMEWORK_DIRECTORIES "")

set(CMAKE_CUDA_IMPLICIT_INCLUDE_DIRECTORIES "/share/apps/scibuilder-spack/aalto-centos7/2023-01/software/linux-centos7-haswell/gcc-11.3.0/openmpi-4.1.5-3qsfx7s/include;/share/apps/scibuilder-spack/aalto-centos7/2023-01/software/linux-centos7-haswell/gcc-11.3.0/ucx-1.13.1-zsom7bz/include;/share/apps/scibuilder-spack/aalto-centos7/2023-01/software/linux-centos7-haswell/gcc-11.3.0/gdrcopy-2.3-zqq5igr/include;/share/apps/scibuilder-spack/aalto-centos7/2023-01/software/linux-centos7-haswell/gcc-11.3.0/pmix-4.2.3-zdmnxjv/include;/share/apps/scibuilder-spack/aalto-centos7/2023-01/software/linux-centos7-haswell/gcc-11.3.0/libevent-2.1.12-lxqv25o/include;/share/apps/scibuilder-spack/aalto-centos7/2023-01/software/linux-centos7-haswell/gcc-11.3.0/libxcrypt-4.4.33-eq5u5e3/include;/share/apps/scibuilder-spack/aalto-centos7/2023-01/software/linux-centos7-haswell/gcc-11.3.0/libedit-3.1-20210216-mqq2hb4/include;/share/apps/scibuilder-spack/aalto-centos7/2023-01/software/linux-centos7-haswell/gcc-11.3.0/numactl-2.0.14-4jrephk/include;/share/apps/scibuilder-spack/aalto-centos7/2023-01/software/linux-centos7-haswell/gcc-11.3.0/hwloc-2.9.1-q7jqcoz/include;/share/apps/scibuilder-spack/aalto-centos7/2023-01/software/linux-centos7-haswell/gcc-11.3.0/libpciaccess-0.17-mlpfrcq/include;/share/apps/scibuilder-spack/aalto-centos7/2023-01/software/linux-centos7-haswell/gcc-11.3.0/libxml2-2.10.3-vkxb5rt/include;/share/apps/scibuilder-spack/aalto-centos7/2023-01/software/linux-centos7-haswell/gcc-11.3.0/xz-5.4.1-la2tl45/include;/share/apps/scibuilder-spack/aalto-centos7/2023-01/software/linux-centos7-haswell/gcc-11.3.0/libiconv-1.17-46m4pm6/include;/share/apps/scibuilder-spack/aalto-centos7/2023-01/software/linux-centos7-haswell/gcc-11.3.0/openssl-1.1.1t-6qqyul4/include;/share/apps/scibuilder-spack/aalto-centos7/2023-01/software/linux-centos7-haswell/gcc-11.3.0/zlib-1.2.13-hbylos3/include;/share/apps/scibuilder-spack/aalto-centos7/2023-01/software/linux-centos7-haswell/gcc-11.3.0/ncurses-6.4-5v4u3ct/include;/share/apps/scibuilder-spack/aalto-centos7/2023-01-compilers/software/linux-centos7-haswell/gcc-4.8.5/gcc-11.3.0-7prhbnn/include/c++/11.3.0;/share/apps/scibuilder-spack/aalto-centos7/2023-01-compilers/software/linux-centos7-haswell/gcc-4.8.5/gcc-11.3.0-7prhbnn/include/c++/11.3.0/x86_64-pc-linux-gnu;/share/apps/scibuilder-spack/aalto-centos7/2023-01-compilers/software/linux-centos7-haswell/gcc-4.8.5/gcc-11.3.0-7prhbnn/include/c++/11.3.0/backward;/share/apps/scibuilder-spack/aalto-centos7/2023-01-compilers/software/linux-centos7-haswell/gcc-4.8.5/gcc-11.3.0-7prhbnn/lib/gcc/x86_64-pc-linux-gnu/11.3.0/include;/usr/local/include;/share/apps/scibuilder-spack/aalto-centos7/2023-01-compilers/software/linux-centos7-haswell/gcc-4.8.5/gcc-11.3.0-7prhbnn/include;/share/apps/scibuilder-spack/aalto-centos7/2023-01-compilers/software/linux-centos7-haswell/gcc-4.8.5/gcc-11.3.0-7prhbnn/lib/gcc/x86_64-pc-linux-gnu/11.3.0/include-fixed;/usr/include")
set(CMAKE_CUDA_IMPLICIT_LINK_LIBRARIES "stdc++;m;gcc_s;gcc;c;gcc_s;gcc")
set(CMAKE_CUDA_IMPLICIT_LINK_DIRECTORIES "/share/apps/scibuilder-spack/aalto-centos7/2023-01/software/linux-centos7-haswell/gcc-11.3.0/cuda-11.8.0-gff3eyf/targets/x86_64-linux/lib/stubs;/share/apps/scibuilder-spack/aalto-centos7/2023-01/software/linux-centos7-haswell/gcc-11.3.0/cuda-11.8.0-gff3eyf/targets/x86_64-linux/lib;/share/apps/scibuilder-spack/aalto-centos7/2023-01/software/linux-centos7-haswell/gcc-11.3.0/gdrcopy-2.3-zqq5igr/lib64;/share/apps/scibuilder-spack/aalto-centos7/2023-01-compilers/software/linux-centos7-haswell/gcc-4.8.5/gcc-11.3.0-7prhbnn/lib64;/share/apps/scibuilder-spack/aalto-centos7/2023-01-compilers/software/linux-centos7-haswell/gcc-4.8.5/gcc-11.3.0-7prhbnn/lib/gcc/x86_64-pc-linux-gnu/11.3.0;/lib64;/usr/lib64;/share/apps/scibuilder-spack/aalto-centos7/2023-01/software/linux-centos7-haswell/gcc-11.3.0/openmpi-4.1.5-3qsfx7s/lib;/share/apps/scibuilder-spack/aalto-centos7/2023-01/software/linux-centos7-haswell/gcc-11.3.0/ucx-1.13.1-zsom7bz/lib;/share/apps/scibuilder-spack/aalto-centos7/2023-01/software/linux-centos7-haswell/gcc-11.3.0/gdrcopy-2.3-zqq5igr/lib;/share/apps/scibuilder-spack/aalto-centos7/2023-01/software/linux-centos7-haswell/gcc-11.3.0/pmix-4.2.3-zdmnxjv/lib;/share/apps/scibuilder-spack/aalto-centos7/2023-01/software/linux-centos7-haswell/gcc-11.3.0/libevent-2.1.12-lxqv25o/lib;/share/apps/scibuilder-spack/aalto-centos7/2023-01/software/linux-centos7-haswell/gcc-11.3.0/libxcrypt-4.4.33-eq5u5e3/lib;/share/apps/scibuilder-spack/aalto-centos7/2023-01/software/linux-centos7-haswell/gcc-11.3.0/libedit-3.1-20210216-mqq2hb4/lib;/share/apps/scibuilder-spack/aalto-centos7/2023-01/software/linux-centos7-haswell/gcc-11.3.0/numactl-2.0.14-4jrephk/lib;/share/apps/scibuilder-spack/aalto-centos7/2023-01/software/linux-centos7-haswell/gcc-11.3.0/hwloc-2.9.1-q7jqcoz/lib;/share/apps/scibuilder-spack/aalto-centos7/2023-01/software/linux-centos7-haswell/gcc-11.3.0/libpciaccess-0.17-mlpfrcq/lib;/share/apps/scibuilder-spack/aalto-centos7/2023-01/software/linux-centos7-haswell/gcc-11.3.0/cuda-11.8.0-gff3eyf/lib64;/share/apps/scibuilder-spack/aalto-centos7/2023-01/software/linux-centos7-haswell/gcc-11.3.0/libxml2-2.10.3-vkxb5rt/lib;/share/apps/scibuilder-spack/aalto-centos7/2023-01/software/linux-centos7-haswell/gcc-11.3.0/xz-5.4.1-la2tl45/lib;/share/apps/scibuilder-spack/aalto-centos7/2023-01/software/linux-centos7-haswell/gcc-11.3.0/libiconv-1.17-46m4pm6/lib;/share/apps/scibuilder-spack/aalto-centos7/2023-01/software/linux-centos7-haswell/gcc-11.3.0/openssl-1.1.1t-6qqyul4/lib;/share/apps/scibuilder-spack/aalto-centos7/2023-01/software/linux-centos7-haswell/gcc-11.3.0/zlib-1.2.13-hbylos3/lib;/share/apps/scibuilder-spack/aalto-centos7/2023-01/software/linux-centos7-haswell/gcc-11.3.0/ncurses-6.4-5v4u3ct/lib;/share/apps/scibuilder-spack/aalto-centos7/2023-01-compilers/software/linux-centos7-haswell/gcc-4.8.5/gcc-11.3.0-7prhbnn/lib")
set(CMAKE_CUDA_IMPLICIT_LINK_FRAMEWORK_DIRECTORIES "")

set(CMAKE_CUDA_RUNTIME_LIBRARY_DEFAULT "STATIC")

set(CMAKE_LINKER "/usr/bin/ld")
set(CMAKE_AR "/usr/bin/ar")
set(CMAKE_MT "")
