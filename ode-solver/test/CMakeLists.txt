add_executable(smatrix_test_app
    test_BlockMatrix_schur3d.cpp
)

# 链接静态库；因为 include 目录是 PUBLIC 的，头文件路径会自动传递过来
target_link_libraries(smatrix_test_app PRIVATE math_lib)