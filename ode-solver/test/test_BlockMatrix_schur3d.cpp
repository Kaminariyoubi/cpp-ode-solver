#include <vector>
#include <cmath>
#include <cstdio>
#include <functional>
#include <algorithm>
#include <cstdint>
#include <memory>
#include <chrono>

#ifdef CPPNUM_HAVE_OPENMP
#include <omp.h>
#endif

#include "math/array.hpp"
#include "math/blockMatrix.hpp"

using namespace sparse;
using namespace block;

// 确定性伪随机数
static double drand(uint32_t i) { 
    double s = std::sin((double)i * 0.371 + 1.7) * 43758.5453; 
    return s - std::floor(s); 
}

// =====================================================================
// 3D MAC 网格 Stokes 算例组装
// =====================================================================
struct Stokes3DData {
    SMatrix K_full;
    SMatrix A_mat;
    SMatrix B_mat;
    SMatrix Bt_mat;
    SMatrix Lp_mat; // 压力 Laplace 矩阵
    uint32_t N_vel;
    uint32_t N_pres;
};

Stokes3DData make_3d_stokes_test(uint32_t N) {
    Stokes3DData data;
    uint32_t N3 = N * N * N;
    data.N_vel = 3 * N3;
    data.N_pres = N3;
    uint32_t N_total = data.N_vel + data.N_pres;

    std::vector<double> vK, vA, vB, vBt, vLp;
    std::vector<uint32_t> rK, cK, rA, cA, rB, cB, rBt, cBt, rLp, cLp;

    auto idx = [&](uint32_t i, uint32_t j, uint32_t k) {
        return i + j * N + k * N * N;
    };

    const double alpha = 0.1; // Helmholtz 正则化参数

    // 1. 组装 A (3 个独立的 3D 标量 Laplace + alpha*I)
    for (uint32_t comp = 0; comp < 3; ++comp) {
        uint32_t offset = comp * N3;
        for (uint32_t k = 0; k < N; ++k) {
            for (uint32_t j = 0; j < N; ++j) {
                for (uint32_t i = 0; i < N; ++i) {
                    uint32_t r = offset + idx(i, j, k);
                    rA.push_back(r); cA.push_back(r); vA.push_back(6.0 + alpha);
                    rK.push_back(r); cK.push_back(r); vK.push_back(6.0 + alpha);
                    
                    uint32_t neighbors[6][3] = {
                        {(i+1)%N, j, k}, {(i-1+N)%N, j, k},
                        {i, (j+1)%N, k}, {i, (j-1+N)%N, k},
                        {i, j, (k+1)%N}, {i, j, (k-1+N)%N}
                    };
                    for (int d = 0; d < 6; ++d) {
                        uint32_t c = offset + idx(neighbors[d][0], neighbors[d][1], neighbors[d][2]);
                        rA.push_back(r); cA.push_back(c); vA.push_back(-1.0);
                        rK.push_back(r); cK.push_back(c); vK.push_back(-1.0);
                    }
                }
            }
        }
    }

    // 2. 组装 B (散度), Bt (负梯度) 和 Lp (压力 Laplace)
    for (uint32_t k = 0; k < N; ++k) {
        for (uint32_t j = 0; j < N; ++j) {
            for (uint32_t i = 0; i < N; ++i) {
                uint32_t p_node = idx(i, j, k); 
                uint32_t r_p = data.N_vel + p_node; 

                // 【关键】Lp 组装 (3D 压力 Laplace + 0.1 扰动)
                rLp.push_back(p_node); cLp.push_back(p_node); vLp.push_back(6.0 + 1e-8);
                uint32_t neighbors[6][3] = {
                    {(i+1)%N, j, k}, {(i-1+N)%N, j, k},
                    {i, (j+1)%N, k}, {i, (j-1+N)%N, k},
                    {i, j, (k+1)%N}, {i, j, (k-1+N)%N}
                };
                for (int d = 0; d < 6; ++d) {
                    uint32_t c_p = idx(neighbors[d][0], neighbors[d][1], neighbors[d][2]);
                    rLp.push_back(p_node); cLp.push_back(c_p); vLp.push_back(-1.0);
                }

                // B, Bt, K 组装
                auto assemble_vel_comp = [&](uint32_t comp, uint32_t curr, uint32_t prev) {
                    uint32_t u_curr = comp * N3 + curr;
                    uint32_t u_prev = comp * N3 + prev;
                    
                    rB.push_back(p_node); cB.push_back(u_curr); vB.push_back(1.0);
                    rB.push_back(p_node); cB.push_back(u_prev); vB.push_back(-1.0);
                    rBt.push_back(u_curr); cBt.push_back(p_node); vBt.push_back(1.0);
                    rBt.push_back(u_prev); cBt.push_back(p_node); vBt.push_back(-1.0);
                    
                    rK.push_back(r_p); cK.push_back(u_curr); vK.push_back(1.0);
                    rK.push_back(r_p); cK.push_back(u_prev); vK.push_back(-1.0);
                    rK.push_back(u_curr); cK.push_back(r_p); vK.push_back(1.0);
                    rK.push_back(u_prev); cK.push_back(r_p); vK.push_back(-1.0);
                };

                assemble_vel_comp(0, p_node, idx((i-1+N)%N, j, k)); 
                assemble_vel_comp(1, p_node, idx(i, (j-1+N)%N, k)); 
                assemble_vel_comp(2, p_node, idx(i, j, (k-1+N)%N)); 
            }
        }
    }

    data.A_mat = SMatrix(data.N_vel, data.N_vel, vA, rA, cA);
    data.B_mat = SMatrix(data.N_pres, data.N_vel, vB, rB, cB);
    data.Bt_mat = SMatrix(data.N_vel, data.N_pres, vBt, rBt, cBt);
    data.Lp_mat = SMatrix(data.N_pres, data.N_pres, vLp, rLp, cLp);
    data.K_full = SMatrix(N_total, N_total, vK, rK, cK);

    return data;
}

// =====================================================================
// 核心测试：GMRES + Schur 补 + Block AMG
// =====================================================================
void test_3d_schur_with_amg(uint32_t N) {
    auto data = make_3d_stokes_test(N);
    uint32_t N_total = data.N_vel + data.N_pres;
    
    // 1. 速度块预处理器 (ILU0)
    auto A_precond = SMatrix::make_ilu0_preconditioner(data.A_mat);
    
    // 2. 压力块预处理器 (Block AMG)
    // 【核心操作】将标量 SMatrix 凝聚为 BS=1 的 BlockMatrix，以适配 Block AMG 接口
    BlockMatrix<double, 1> Lp_blk = BlockMatrix<double, 1>::condense(
        data.Lp_mat, data.Lp_mat.rows(), data.Lp_mat.cols());
    
    auto Lp_amg = make_block_amg_preconditioner(Lp_blk);
    
    // 3. 构造最优 Schur 补预处理器: S_pre^{-1} = I + alpha * Lp^{-1}
    const double alpha = 0.1; // 必须与组装 A 和 Lp 时的 alpha 严格一致
    auto S_pre_amg = [Lp_amg](const std::vector<double>& r, std::vector<double>& z) 
    {
        std::vector<double> w;
        Lp_amg(r, w);  // AMG 求解 L_p w = r
        
        z.resize(r.size());
        const double scale = 6.1; // A 的对角元 (6.0 + 0.1)
        for (size_t i = 0; i < r.size(); ++i) {
            z[i] = scale * w[i];  // 正确的物理近似：S^{-1} \approx 6.1 * L_p^{-1}
        }
    };
    
    // Baseline: S_pre = I
    auto S_pre_I = [](const std::vector<double>& r, std::vector<double>& z) { z = r; };
    
    // 构造鞍点系统接口 (值捕获防止悬挂引用)
    SaddlePointSystem sys;
    sys.n_vel = data.N_vel;
    sys.n_pres = data.N_pres;
    SMatrix A_copy = data.A_mat, B_copy = data.B_mat, Bt_copy = data.Bt_mat;
    sys.spmv_A  = [A_copy](const std::vector<double>& x, std::vector<double>& y) { y = SMatrix::spmv(A_copy, x); };
    sys.spmv_B  = [B_copy](const std::vector<double>& x, std::vector<double>& y) { y = SMatrix::spmv(B_copy, x); };
    sys.spmv_Bt = [Bt_copy](const std::vector<double>& x, std::vector<double>& y) { y = SMatrix::spmv(Bt_copy, x); };
    
    // 构造右端项
    std::vector<double> x_exact(N_total), b;
    for(uint32_t i = 0; i < N_total; ++i) x_exact[i] = drand(i);
    b = SMatrix::spmv(data.K_full, x_exact);
    
    // 测试 1: S_pre = I (Baseline)
    auto schur_I = make_schur_preconditioner(sys, A_precond, S_pre_I);
    auto [x1, it1] = SMatrix::gmres(data.K_full, b, schur_I, 100);
    
    // 测试 2: S_pre = I + alpha * AMG(Lp)^{-1} (最优)
    auto schur_amg = make_schur_preconditioner(sys, A_precond, S_pre_amg);
    auto [x2, it2] = SMatrix::gmres(data.K_full, b, schur_amg, 100);
    
    double err2 = 0.0;
    for(uint32_t i = 0; i < N_total; ++i) err2 += (x2[i]-x_exact[i])*(x2[i]-x_exact[i]);
    
    std::printf("[3D Stokes+AMG] N=%3u DOF=%7u | S_pre=I: %4d步 | S_pre=AMG: %4d步 残差=%.2e %s\n",
                N, N_total, it1+1, it2+1, std::sqrt(err2),
                std::sqrt(err2) < 1e-6 ? "通过" : "失败");
}

int main() {
    // 测试不同网格分辨率，观察 Mesh-independence (网格无关性)

    std::vector<uint32_t> Nlist = {50, 75};

    for (uint32_t i = 0; i < Nlist.size(); i++)
    {
        std::cout << "=== gmres + amg + 舒尔补 + 三维斯托克斯 (N = " << Nlist[i] << ") ===\n";
        auto start_serial = std::chrono::high_resolution_clock::now();
        test_3d_schur_with_amg(Nlist[i]);
        auto end_serial = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> time_serial = end_serial - start_serial;
        std::cout << "N =" << Nlist[i] << "：" << "耗时" << time_serial.count() << " s\n";
    }
    return 0;
}