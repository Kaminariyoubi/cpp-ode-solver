# ============== 寻找 MKL =================

set(MKLROOT "$ENV{CONDA_PREFIX}")
if(NOT MKLROOT)
    set(MKLROOT "/home/kazeyoubi/miniconda3/envs/cfd7")
endif()

find_path(MKL_INCLUDE_DIR mkl.h PATHS ${MKLROOT}/include NO_DEFAULT_PATH)
find_library(MKL_RT_LIB mkl_rt PATHS ${MKLROOT}/lib NO_DEFAULT_PATH)

if(NOT MKL_INCLUDE_DIR OR NOT MKL_RT_LIB)
    message(FATAL_ERROR "MKL not found in ${MKLROOT}")
endif()

message(STATUS "[src] MKL_INCLUDE_DIR = ${MKL_INCLUDE_DIR}")
message(STATUS "[src] MKL_RT_LIB = ${MKL_RT_LIB}")

# ============== 编译静态库 =================

# 把 src 下的源码编译成静态库
add_library(math_lib STATIC
    math/integrate.cpp
    math/array.cpp
)

# =============== 链接 include ===============

# 把项目根目录的 include/ 作为 PUBLIC 头文件目录暴露出去：
# - integrate.cpp 自己可以 #include "math/integral.hpp"
# - 链接了 math_lib 的目标(如 main.cpp)也自动获得该头文件路径

target_include_directories(math_lib PUBLIC
    ${PROJECT_SOURCE_DIR}/include
    ${MKL_INCLUDE_DIRS}
)

target_link_libraries(math_lib PUBLIC cppnum_warnings)
target_link_libraries(math_lib PUBLIC ${MKL_RT_LIB})

# =============== 链接并行库 ===================

if(CPPNUM_ENABLE_OPENMP)
    target_link_libraries(math_lib PUBLIC OpenMP::OpenMP_CXX)   # 真正的编译/链接旗标
    target_compile_definitions(math_lib PUBLIC CPPNUM_HAVE_OPENMP)  # 源码里可用的宏
endif()

if(CPPNUM_ENABLE_MPI)
    target_link_libraries(math_lib PUBLIC MPI::MPI_CXX)
    target_compile_definitions(math_lib PUBLIC CPPNUM_HAVE_MPI)
endif()

