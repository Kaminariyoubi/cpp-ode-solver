#include <vector>
#include <cmath>
#include <cstdio>
#include <functional>
#include <algorithm>
#include <cstdint>
#ifdef CPPNUM_HAVE_OPENMP
#include <omp.h>
#endif
#include "math/array.hpp"
#include "math/blockMatrix.hpp"

using namespace sparse;
using namespace block;

static double drand(uint32_t i)
{ 
    double s = std::sin((double)i * 0.371 + 1.7) * 43758.5453; 
    return s - std::floor(s); 
}

// =====================================================================
// 1D Stokes 测试数据
// =====================================================================
struct StokesTestData {
    SMatrix K_full;
    SMatrix A_mat;
    SMatrix B_mat;      // 【修复】存入结构体，延长生命周期
    SMatrix Bt_mat;     // 【修复】存入结构体，延长生命周期
    SaddlePointSystem sys;
};

StokesTestData make_1d_stokes_test(uint32_t N_vel) 
{
    StokesTestData data;
    uint32_t N_pres = N_vel - 1;
    uint32_t N_total = N_vel + N_pres;

    std::vector<double> vals; std::vector<uint32_t> rl, cl;
    
    for (uint32_t i = 0; i < N_vel; ++i) {
        rl.push_back(i); cl.push_back(i); vals.push_back(2.0);
        if (i > 0) { rl.push_back(i); cl.push_back(i-1); vals.push_back(-1.0); }
        if (i < N_vel-1) { rl.push_back(i); cl.push_back(i+1); vals.push_back(-1.0); }
    }
    
    for (uint32_t i = 0; i < N_pres; ++i) {
        rl.push_back(i + N_vel); cl.push_back(i); vals.push_back(-1.0);
        rl.push_back(i + N_vel); cl.push_back(i + 1); vals.push_back(1.0);
        rl.push_back(i); cl.push_back(i + N_vel); vals.push_back(-1.0);
        rl.push_back(i + 1); cl.push_back(i + N_vel); vals.push_back(1.0);
    }

    data.K_full = SMatrix(N_total, N_total, vals, rl, cl);
    
    std::vector<double> vA; std::vector<uint32_t> rA, cA;
    std::vector<double> vB; std::vector<uint32_t> rB, cB;
    std::vector<double> vBt; std::vector<uint32_t> rBt, cBt;
    
    for(uint32_t k = 0; k < vals.size(); ++k) {
        if(rl[k] < N_vel && cl[k] < N_vel)   { rA.push_back(rl[k]); cA.push_back(cl[k]); vA.push_back(vals[k]); }
        if(rl[k] >= N_vel && cl[k] < N_vel)  { rB.push_back(rl[k]-N_vel); cB.push_back(cl[k]); vB.push_back(vals[k]); }
        if(rl[k] < N_vel && cl[k] >= N_vel)  { rBt.push_back(rl[k]); cBt.push_back(cl[k]-N_vel); vBt.push_back(vals[k]); }
    }
    
    data.A_mat  = SMatrix(N_vel, N_vel, vA, rA, cA);
    data.B_mat  = SMatrix(N_pres, N_vel, vB, rB, cB);       // 存入结构体
    data.Bt_mat = SMatrix(N_vel, N_pres, vBt, rBt, cBt);    // 存入结构体
    
    data.sys.n_vel = N_vel;
    data.sys.n_pres = N_pres;
    // 【修复】值捕获 data.A_mat / data.B_mat / data.Bt_mat（它们在 data 中，生命周期足够）
    data.sys.spmv_A  = [&data](const std::vector<double>& x, std::vector<double>& y) { y = SMatrix::spmv(data.A_mat, x); };
    data.sys.spmv_B  = [&data](const std::vector<double>& x, std::vector<double>& y) { y = SMatrix::spmv(data.B_mat, x); };
    data.sys.spmv_Bt = [&data](const std::vector<double>& x, std::vector<double>& y) { y = SMatrix::spmv(data.Bt_mat, x); };
    
    return data;
}

// =====================================================================
// 测试：GMRES + Schur 补
// =====================================================================
void test_schur_complement(uint32_t N_vel) {
    auto data = make_1d_stokes_test(N_vel);
    uint32_t N_total = N_vel + data.sys.n_pres;
    
    // A 的预处理器 (标量 ILU0)
    auto A_precond = SMatrix::make_ilu0_preconditioner(data.A_mat);

    std::cout << "ilu预处理器加载完成" << std::endl;
    
    // S 的预处理器 (简化: 单位阵)
    auto S_precond = [](const std::vector<double>& r, std::vector<double>& z) { z = r; };
    
    // Schur 预处理器
    auto schur_precond = make_schur_preconditioner(data.sys, A_precond, S_precond);

    std::cout << "舒尔补预处理器加载完成" << std::endl;
    
    // 右端项
    std::vector<double> x_exact(N_total), b;
    for(uint32_t i = 0; i < N_total; ++i) x_exact[i] = drand(i);
    b = SMatrix::spmv(data.K_full, x_exact);
    
    // 【关键】调用 SMatrix::gmres 的 std::function 重载
    auto [x, it] = SMatrix::gmres(data.K_full, b, schur_precond, 50);
    
    std::cout << "gmres求解完成" << std::endl;

    double err = 0.0;
    for(uint32_t i = 0; i < N_total; ++i) err += (x[i] - x_exact[i]) * (x[i] - x_exact[i]);
    
    std::printf("[Schur] N_vel=%4u GMRES迭代数=%4d 相对误差=%.3e %s\n",
                N_vel, it+1, std::sqrt(err), std::sqrt(err) < 1e-6 ? "通过" : "失败");
}

int main()
{
    test_schur_complement(100);
    test_schur_complement(500);
    test_schur_complement(1000);
    return 0;
}