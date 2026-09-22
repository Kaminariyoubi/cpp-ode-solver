#pragma once
#include <vector>
#include <array>
#include <cstdint>
#include <algorithm>
#include <map>
#ifdef CPPNUM_HAVE_OPENMP
#include <omp.h>
#endif
#ifdef CPPNUM_HAVE_MPI
#include <mpi.h>
#endif
#include "math/array.hpp"   // SMatrix

using namespace sparse;

namespace block 
{

    #ifdef CPPNUM_HAVE_MPI

    // MPI 数据类型映射 Traits
    template <typename T> struct MPI_Type_Traits;
    template <> struct MPI_Type_Traits<double> { static MPI_Datatype type() { return MPI_DOUBLE; } };
    template <> struct MPI_Type_Traits<float>  { static MPI_Datatype type() { return MPI_FLOAT; } };
    template <> struct MPI_Type_Traits<int>    { static MPI_Datatype type() { return MPI_INT; } };
    template <> struct MPI_Type_Traits<unsigned int> { static MPI_Datatype type() { return MPI_UNSIGNED; } };

    // 便捷调用函数
    template <typename T>
    inline MPI_Datatype mpi_type() 
    { 
        return MPI_Type_Traits<T>::type(); 
    }
    #endif

    template <int BS>
    inline void blk_matmat_accum(const double* A, const double* B, double* C)
    {
        for (int r = 0; r < BS; ++r)
            for (int c = 0; c < BS; ++c)
            {
                double s = 0.0;
                for (int k = 0; k < BS; ++k) s += A[r*BS+k] * B[k*BS+c];
                C[r*BS+c] -= s;
            }
    }
    // C := C*B   (右乘, 不可交换性的体现)
    template <int BS>
    inline void blk_right_mul(double* C, const double* B)
    {
        double tmp[BS*BS];
        for (int r = 0; r < BS; ++r)
            for (int c = 0; c < BS; ++c)
            {
                double s = 0.0;
                for (int k = 0; k < BS; ++k) s += C[r*BS+k] * B[k*BS+c];
                tmp[r*BS+c] = s;
            }
        for (int t = 0; t < BS*BS; ++t) C[t] = tmp[t];
    }
    // y = M*x
    template <int BS>
    inline void blk_matvec(const double* M, const double* x, double* y)
    {
        for (int r = 0; r < BS; ++r)
        {
            double s = 0.0;
            for (int c = 0; c < BS; ++c) s += M[r*BS+c] * x[c];
            y[r] = s;
        }
    }
    // Minv = M^{-1} (高斯-部分主元); 失败返回 false
    template <int BS>
    inline bool blk_invert(const double* M, double* Minv)
    {
        double aug[BS][2*BS];
        for (int i = 0; i < BS; ++i)
            for (int j = 0; j < BS; ++j)
            { aug[i][j] = M[i*BS+j]; aug[i][BS+j] = (i==j) ? 1.0 : 0.0; }
        for (int k = 0; k < BS; ++k)
        {
            int piv = k;
            for (int i = k+1; i < BS; ++i)
                if (std::fabs(aug[i][k]) > std::fabs(aug[piv][k])) piv = i;
            if (std::fabs(aug[piv][k]) < 1e-300) return false;
            if (piv != k)
                for (int j = 0; j < 2*BS; ++j) std::swap(aug[k][j], aug[piv][j]);
            for (int i = k+1; i < BS; ++i)
            {
                const double f = aug[i][k] / aug[k][k];
                for (int j = k; j < 2*BS; ++j) aug[i][j] -= f * aug[k][j];
            }
        }
        for (int i = BS-1; i >= 0; --i)
        {
            for (int j = i+1; j < BS; ++j)
                for (int c = 0; c < BS; ++c) aug[i][BS+c] -= aug[i][j] * aug[j][BS+c];
            for (int c = 0; c < BS; ++c) aug[i][BS+c] /= aug[i][i];
        }
        for (int i = 0; i < BS; ++i)
            for (int j = 0; j < BS; ++j) Minv[i*BS+j] = aug[i][BS+j];
        return true;
    }
    // 行内二分查找块列; 未找到返回 UINT32_MAX (行内块列升序前提)
    inline uint32_t blk_find_col(const std::vector<uint32_t>& rp,
                                const std::vector<uint32_t>& ci,
                                uint32_t row, uint32_t col)
    {
        uint32_t lo = rp[row], hi = rp[row+1];
        while (lo < hi)
        {
            const uint32_t mid = lo + (hi - lo) / 2;
            if (ci[mid] == col) return mid;
            if (ci[mid] < col) lo = mid + 1; else hi = mid;
        }
        return UINT32_MAX;
    }

    // Block-ILU(0) 数据: L (单位块下三角, 对角隐含), U (块上三角含对角), inv(U_KK)
    template <int BS>
    struct BlockILU0_Data
    {
        std::vector<uint32_t> L_row_ptr, L_col_idx;
        std::vector<double>   L_values;
        std::vector<uint32_t> U_row_ptr, U_col_idx;
        std::vector<double>   U_values;
        std::vector<uint32_t> U_diag_ptr;    // 每块行对角块的块偏移
        std::vector<double>   inv_diag;      // nb * BS*BS
        mutable std::vector<double> y_buf;   // apply 工作缓冲
    };
// =====================================================================
// BlockMatrix<T, BS> : Block-CSR (BAIJ) 分布式块稀疏矩阵
//   T  : 标量类型 (double)
//   BS : 块边长 (编译期: 2D NS 同阶元=3, 3D=4)
// 向量布局: node-major (块内 BS 个分量连续), 与 FEM 组装自然顺序一致,
//           因此 std::vector<double> 块向量可零改动接入现有 Krylov 求解器
// =====================================================================
    template <typename T, int BS>
    class BlockMatrix
    {
        private:
            uint32_t nbr_, nbc_;
            std::vector<uint32_t> row_ptr_;      // 块行指针 (长度 nbr+1)
            std::vector<uint32_t> col_idx_;      // 块列索引 (全局编号)
            std::vector<T>        values_;       // 连续块值 (nnz_blocks * BS*BS)
            std::vector<uint32_t> diag_ptr_;     // 每块行对角块的块偏移 (UINT32_MAX=缺)

            // ---- MPI 成员 ----
            int      rank_ = 0, nprocs_ = 1;
            uint32_t n_owned_ = 0, n_ghost_ = 0;          // 本地 owned / ghost 块行数
            std::vector<uint32_t> local_col_idx_;         // 全局块列 -> x_full 本地索引
            std::vector<int>      recv_procs_, recv_cnts_;
            std::vector<uint32_t> recv_global_;           // 要接收的全局块行
            std::vector<int>      send_procs_, send_cnts_;
            std::vector<uint32_t> send_local_;            // x_owned 中要发送的索引
            mutable std::vector<T> x_full_;               // owned+ghost 缓冲 (预分配)
            mutable std::vector<T> send_buf_;
            #ifdef CPPNUM_HAVE_MPI
                mutable std::vector<MPI_Request> reqs_;
            #endif


        public:
            static constexpr int kBlock     = BS;
            static constexpr int kBlockVals = BS * BS;

            // 构造函数
            BlockMatrix() : nbr_(0), nbc_(0), row_ptr_{0} {}
            BlockMatrix(
                uint32_t nbr, uint32_t nbc, 
                std::vector<uint32_t> row_ptr,
                std::vector<uint32_t> col_idx,
                std::vector<T> values
            ) : nbr_(nbr), nbc_(nbc),
                row_ptr_(std::move(row_ptr)),
                col_idx_(std::move(col_idx)),
                values_(std::move(values))
            { 
                cache_diag(); 
            }
            
            // ---- 访问器 ----
            uint32_t block_rows() const { return nbr_; }
            uint32_t block_cols() const { return nbc_; }
            uint32_t rows() const { return nbr_ * BS; }      // 标量自由度行数
            uint32_t cols() const { return nbc_ * BS; }
            size_t   nnz_blocks() const { return col_idx_.size(); }
            size_t   nnz() const { return col_idx_.size() * (size_t)kBlockVals; }

            const std::vector<uint32_t>& block_row_ptr() const { return row_ptr_; }
            const std::vector<uint32_t>& block_col_idx() const { return col_idx_; }
            const std::vector<T>&        values()        const { return values_; }

            const T* block(uint32_t p) const { return &values_[(size_t)p * kBlockVals]; }
            const T* diag_block(uint32_t bi) const { return &values_[(size_t)diag_ptr_[bi] * kBlockVals]; }

            // ---- 核心运算 ----
            void cache_diag()
            {
                diag_ptr_.assign(nbr_, UINT32_MAX);
                for (uint32_t bi = 0; bi < nbr_; ++bi)
                    for (uint32_t p = row_ptr_[bi]; p < row_ptr_[bi + 1]; ++p)
                        if (col_idx_[p] == bi) { diag_ptr_[bi] = p; break; }
            }

            // 矩阵向量乘法
            void matvec(const std::vector<T>& x, std::vector<T>& y) const
            {
                y.assign(rows(), T(0));
                #ifdef CPPNUM_HAVE_OPENMP
                    #pragma omp parallel for schedule(static)
                #endif
                for (std::ptrdiff_t bi = 0; bi < (std::ptrdiff_t)nbr_; ++bi)
                {
                    T* yi = &y[(size_t)bi * BS];
                    for (uint32_t p = row_ptr_[bi]; p < row_ptr_[bi + 1]; ++p)
                    {
                        const T* B  = block(p);
                        const T* xj = &x[(size_t)col_idx_[p] * BS];
                        // 块内 GEMV: yi += B * xj  (BS 编译期展开 + simd)
                        for (int r = 0; r < BS; ++r)
                        {
                            T acc = T(0);
                            #ifdef CPPNUM_HAVE_OPENMP
                                #pragma omp simd reduction(+:acc)
                            #endif
                            for (int c = 0; c < BS; ++c)
                                acc += B[r * BS + c] * xj[c];
                            yi[r] += acc;
                        }
                    }
                }
            }

            // 转置
            BlockMatrix<T, BS> transpose() const
            {
                // 把块当"标量"做 CSR 转置, 得到块转置的结构
                std::vector<uint32_t> trp(nbc_ + 1, 0);
                for (size_t p = 0; p < col_idx_.size(); ++p) trp[col_idx_[p] + 1]++;
                for (uint32_t i = 0; i < nbc_; ++i) trp[i + 1] += trp[i];
                std::vector<uint32_t> tci(col_idx_.size());
                std::vector<uint32_t> off = trp;
                for (uint32_t i = 0; i < nbr_; ++i)
                    for (uint32_t p = row_ptr_[i]; p < row_ptr_[i + 1]; ++p)
                        tci[off[col_idx_[p]]++] = i;

                std::vector<T> tv(values_.size());
                #ifdef CPPNUM_HAVE_OPENMP
                    #pragma omp parallel for schedule(static)
                #endif
                for (std::ptrdiff_t j = 0; j < (std::ptrdiff_t)nbc_; ++j)      // 转置后的块行
                    for (uint32_t q = trp[j]; q < trp[j + 1]; ++q)
                    {
                        const uint32_t i = tci[q];                              // 原块行
                        // 找原矩阵中块 (i, j) 的位置 p
                        uint32_t p = row_ptr_[i];
                        while (col_idx_[p] != j) ++p;
                        const T* B = block(p);
                        T* C = &tv[(size_t)q * kBlockVals];
                        for (int r = 0; r < BS; ++r)
                            for (int c = 0; c < BS; ++c)
                                C[r * BS + c] = B[c * BS + r];                  // 块内转置
                    }
                return BlockMatrix(nbc_, nbr_, std::move(trp), std::move(tci), std::move(tv));
            }

            static BlockMatrix<T, BS> condense(const SMatrix& A, uint32_t nbr, uint32_t nbc)
            {
                const auto& Ap = A.row_ptr(); const auto& Ac = A.col_idx(); const auto& Av = A.values();

                // Pass 1: 每块行收集去重块列 (行并行, 局部去重)
                std::vector<std::vector<uint32_t>> blkcols(nbr);
                #ifdef CPPNUM_HAVE_OPENMP
                    #pragma omp parallel for schedule(dynamic, 64)
                #endif
                for (std::ptrdiff_t bi = 0; bi < (std::ptrdiff_t)nbr; ++bi)
                {
                    auto& out = blkcols[bi];
                    for (uint32_t i = bi * BS; i < (bi + 1) * BS; ++i)
                        for (uint32_t p = Ap[i]; p < Ap[i + 1]; ++p)
                            out.push_back(Ac[p] / BS);
                    std::sort(out.begin(), out.end());
                    out.erase(std::unique(out.begin(), out.end()), out.end());
                }

                std::vector<uint32_t> rp(nbr + 1, 0);
                for (uint32_t bi = 0; bi < nbr; ++bi) rp[bi + 1] = blkcols[bi].size();
                for (uint32_t bi = 0; bi < nbr; ++bi) rp[bi + 1] += rp[bi];
                std::vector<uint32_t> ci(rp[nbr]);
                std::vector<T> vv((size_t)rp[nbr] * kBlockVals, T(0));

                // Pass 2: 填值 (行并行, 各块行写自己的块值区间, 无竞争)
                #ifdef CPPNUM_HAVE_OPENMP
                    #pragma omp parallel for schedule(dynamic, 64)
                #endif
                for (std::ptrdiff_t bi = 0; bi < (std::ptrdiff_t)nbr; ++bi)
                {
                    for (uint32_t k = 0; k < blkcols[bi].size(); ++k) ci[rp[bi] + k] = blkcols[bi][k];
                    for (uint32_t i = bi * BS; i < (bi + 1) * BS; ++i)
                    {
                        const int r = i % BS;
                        for (uint32_t p = Ap[i]; p < Ap[i + 1]; ++p)
                        {
                            const uint32_t bj = Ac[p] / BS, c = Ac[p] % BS;
                            // 块行 bi 内定位块 bj 的偏移 (块数小, 二分)
                            auto it = std::lower_bound(blkcols[bi].begin(), blkcols[bi].end(), bj);
                            const uint32_t boff = rp[bi] + (uint32_t)(it - blkcols[bi].begin());
                            vv[(size_t)boff * kBlockVals + (size_t)r * BS + c] += Av[p];
                        }
                    }
                }
                return BlockMatrix(nbr, nbc, std::move(rp), std::move(ci), std::move(vv));
            }

            // ---- MPI 分布 ----
            void set_halo(std::vector<int> recv_procs, std::vector<int> recv_cnts,
                  std::vector<uint32_t> recv_global,
                  std::vector<int> send_procs, std::vector<int> send_cnts,
                  std::vector<uint32_t> send_local,
                  uint32_t owned_global_start = 0)
            {
                recv_procs_ = std::move(recv_procs); recv_cnts_ = std::move(recv_cnts);
                recv_global_ = std::move(recv_global);
                send_procs_ = std::move(send_procs); send_cnts_ = std::move(send_cnts);
                send_local_ = std::move(send_local);
                n_ghost_ = (uint32_t)recv_global_.size();
                n_owned_ = nbr_;                                  // 本地行即 owned 行

                // 全局块列 -> x_full 本地索引 (owned 区 + ghost 区), 预计算避免每步查表
                local_col_idx_.resize(col_idx_.size());
                std::vector<uint32_t> ghost_pos(nbc_, UINT32_MAX);
                for (uint32_t k = 0; k < n_ghost_; ++k) ghost_pos[recv_global_[k]] = n_owned_ + k;
                for (size_t p = 0; p < col_idx_.size(); ++p)
                {
                    const uint32_t g = col_idx_[p];
                    if (g >= owned_global_start && g < owned_global_start + n_owned_)
                        local_col_idx_[p] = g - owned_global_start;          // owned 列 -> 本地偏移
                    else
                        local_col_idx_[p] = ghost_pos[g];                    // ghost 列 -> ghost 区
                }
                x_full_.resize(((size_t)n_owned_ + n_ghost_) * BS);
                send_buf_.resize(send_local_.size() * BS);
                #ifdef CPPNUM_HAVE_MPI
                    reqs_.resize(2 * (recv_procs_.size() + send_procs_.size()) + 4);
                #endif
            }
            uint32_t owned_rows() const { return n_owned_; }

            void distributed_matvec(const std::vector<T>& x_owned, std::vector<T>& y_owned) const
            {
                #ifdef CPPNUM_HAVE_MPI
                    int my_rank;
                    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank); // 获取当前进程的 Rank
                    
                    const size_t owned_vals = (size_t)n_owned_ * BS;
                    
                    // 1. 拷贝 owned 区 + 打包 send
                    for (size_t i = 0; i < owned_vals; ++i) x_full_[i] = x_owned[i];
                    for (size_t i = 0; i < send_local_.size(); ++i)
                        for (int c = 0; c < BS; ++c)
                            send_buf_[i * BS + c] = x_owned[(size_t)send_local_[i] * BS + c];

                    // 2. 非阻塞启动通信
                    int rq = 0; 
                    size_t off = owned_vals;
                    
                    // 接收 Ghost 数据
                    for (size_t k = 0; k < recv_procs_.size(); ++k)
                    {
                        int src = recv_procs_[k]; // 源进程的 Rank
                        // 【核心】接收方 Tag = 1000 + 源Rank
                        MPI_Irecv(&x_full_[off], recv_cnts_[k] * BS, mpi_type<T>(),
                                src, 1000 + src, MPI_COMM_WORLD, &reqs_[rq++]);
                        off += (size_t)recv_cnts_[k] * BS;
                    }
                    
                    off = 0;
                    // 发送 Owned 数据
                    for (size_t k = 0; k < send_procs_.size(); ++k)
                    {
                        int dst = send_procs_[k]; // 目标进程的 Rank
                        // 【核心】发送方 Tag = 1000 + 自己的Rank (即源Rank)
                        // 这样无论对方是谁，双方算出的 Tag 永远一致！
                        MPI_Isend(&send_buf_[off], send_cnts_[k] * BS, mpi_type<T>(),
                                dst, 1000 + my_rank, MPI_COMM_WORLD, &reqs_[rq++]);
                        off += (size_t)send_cnts_[k] * BS;
                    }

                    // 3. 通信-计算重叠: 算 owned 列
                    y_owned.assign(owned_vals, T(0));
                    #ifdef CPPNUM_HAVE_OPENMP
                        #pragma omp parallel for schedule(static)
                    #endif
                    for (std::ptrdiff_t bi = 0; bi < (std::ptrdiff_t)n_owned_; ++bi)
                    {
                        T* yi = &y_owned[(size_t)bi * BS];
                        for (uint32_t p = row_ptr_[bi]; p < row_ptr_[bi + 1]; ++p)
                        {
                            if (local_col_idx_[p] >= n_owned_) continue;
                            const T* B = block(p);
                            const T* xj = &x_full_[(size_t)local_col_idx_[p] * BS];
                            for (int r = 0; r < BS; ++r)
                            {
                                T acc = 0;
                                for (int c = 0; c < BS; ++c) acc += B[r * BS + c] * xj[c];
                                yi[r] += acc;
                            }
                        }
                    }

                    // 4. 等待通信完成 (防御性判断)
                    if (rq > 0) 
                    {
                        MPI_Waitall(rq, reqs_.data(), MPI_STATUSES_IGNORE);
                    }

                    // 5. 补算 ghost 列
                    #ifdef CPPNUM_HAVE_OPENMP
                        #pragma omp parallel for schedule(static)
                    #endif
                    for (std::ptrdiff_t bi = 0; bi < (std::ptrdiff_t)n_owned_; ++bi)
                    {
                        T* yi = &y_owned[(size_t)bi * BS];
                        for (uint32_t p = row_ptr_[bi]; p < row_ptr_[bi + 1]; ++p)
                        {
                            if (local_col_idx_[p] < n_owned_) continue;
                            const T* B = block(p);
                            const T* xj = &x_full_[(size_t)local_col_idx_[p] * BS];
                            for (int r = 0; r < BS; ++r)
                            {
                                T acc = 0;
                                for (int c = 0; c < BS; ++c) acc += B[r * BS + c] * xj[c];
                                yi[r] += acc;
                            }
                        }
                    }
                #else
                    matvec(x_owned, y_owned);
                #endif
            }

            static std::tuple<std::vector<T>, int> gmres(
                const BlockMatrix& A,
                const std::vector<T>& b,
                const std::string& precond,
                double tol,
                int max_iter,
                int restart
            );
    };

    // ---- Block-ilu 算法 ----
    template <typename T, int BS>
    std::function<void(const std::vector<T>&, std::vector<T>&)> make_block_ilu0_preconditioner(const BlockMatrix<T, BS>& A) // 【修复1】正确的双模板参数
    {
        if (A.block_rows() != A.block_cols())
            throw std::invalid_argument("Block-ILU(0) 要求块矩阵为方阵");

        const uint32_t nb = A.block_rows();
        const int BV = BS * BS;
        const auto& Arp = A.block_row_ptr();
        const auto& Aci = A.block_col_idx();
        const auto& Avv = A.values();

        // 【修复2】必须加上 <BS>
        auto data = std::make_shared<BlockILU0_Data<BS>>();

        // 1. 计数: L (J<I) 与 U (J>=I)
        data->L_row_ptr.assign(nb + 1, 0);
        data->U_row_ptr.assign(nb + 1, 0);
        #ifdef CPPNUM_HAVE_OPENMP
            #pragma omp parallel for schedule(static)
        #endif
        for (std::ptrdiff_t I = 0; I < (std::ptrdiff_t)nb; ++I)
        {
            uint32_t cL = 0, cU = 0;
            for (uint32_t p = Arp[I]; p < Arp[I+1]; ++p)
            { if (Aci[p] < (uint32_t)I) ++cL; else ++cU; }
            data->L_row_ptr[I+1] = cL;
            data->U_row_ptr[I+1] = cU;
        }
        for (uint32_t I = 0; I < nb; ++I)
        { data->L_row_ptr[I+1] += data->L_row_ptr[I];
        data->U_row_ptr[I+1] += data->U_row_ptr[I]; }

        data->L_col_idx.resize(data->L_row_ptr[nb]);
        data->L_values.resize((size_t)data->L_row_ptr[nb] * BV);
        data->U_col_idx.resize(data->U_row_ptr[nb]);
        data->U_values.resize((size_t)data->U_row_ptr[nb] * BV);
        data->U_diag_ptr.assign(nb, UINT32_MAX);
        std::vector<char> has_diag(nb, 0);

        // 2. 填充
        // 【修复3】补全 offU
        std::vector<uint32_t> offL = data->L_row_ptr, offU = data->U_row_ptr;
        #ifdef CPPNUM_HAVE_OPENMP
            #pragma omp parallel for schedule(static)
        #endif
        for (std::ptrdiff_t I = 0; I < (std::ptrdiff_t)nb; ++I)
        {
            for (uint32_t p = Arp[I]; p < Arp[I+1]; ++p)
            {
                const uint32_t J = Aci[p];
                const T* src = &Avv[(size_t)p * BV];
                if (J < (uint32_t)I)
                {
                    const uint32_t d = offL[I]++;
                    data->L_col_idx[d] = J;
                    std::copy(src, src + BV, &data->L_values[(size_t)d * BV]);
                }
                else
                {
                    const uint32_t d = offU[I]++;
                    data->U_col_idx[d] = J;
                    std::copy(src, src + BV, &data->U_values[(size_t)d * BV]);
                    if (J == (uint32_t)I) { data->U_diag_ptr[I] = d; has_diag[I] = 1; }
                }
            }
        }
        for (uint32_t I = 0; I < nb; ++I)
            if (!has_diag[I])
                throw std::runtime_error("Block-ILU(0) 失败：块行 " + std::to_string(I) + " 缺少对角块");

        // 3. 分解
        data->inv_diag.resize((size_t)nb * BV);
        for (uint32_t I = 0; I < nb; ++I)
        {
            const uint32_t Ls = data->L_row_ptr[I];
            const uint32_t Le = data->L_row_ptr[I+1]; // 【修复4】补全 Le
            const uint32_t Us = data->U_row_ptr[I];
            const uint32_t Ue = data->U_row_ptr[I+1]; // 【修复5】补全 Ue

            for (uint32_t pL = Ls; pL < Le; ++pL)
            {
                const uint32_t K = data->L_col_idx[pL];
                T* LIK = &data->L_values[(size_t)pL * BV];

                // 左和式
                for (uint32_t q = Ls; q < pL; ++q)
                {
                    const uint32_t J = data->L_col_idx[q];
                    const uint32_t pJK = blk_find_col(data->U_row_ptr, data->U_col_idx, J, K);
                    if (pJK != UINT32_MAX)
                        blk_matmat_accum<BS>(&data->L_values[(size_t)q * BV],
                                            &data->U_values[(size_t)pJK * BV], LIK);
                }
                
                // 右乘逆
                blk_right_mul<BS>(LIK, &data->inv_diag[(size_t)K * BV]);

                // 双指针更新 Urow(I)
                uint32_t pi = Us, pk = data->U_row_ptr[K];
                const uint32_t ek = data->U_row_ptr[K+1];
                while (pi < Ue && pk < ek)
                {
                    if (data->U_col_idx[pi] == data->U_col_idx[pk])
                    {
                        blk_matmat_accum<BS>(LIK, &data->U_values[(size_t)pk * BV],
                                            &data->U_values[(size_t)pi * BV]);
                        ++pi; ++pk;
                    }
                    else if (data->U_col_idx[pi] < data->U_col_idx[pk]) ++pi;
                    else ++pk;
                }
            }

            // 缓存对角块之逆
            if (!(blk_invert<BS>(&data->U_values[(size_t)data->U_diag_ptr[I] * BV],
                                &data->inv_diag[(size_t)I * BV])))
                throw std::runtime_error("Block-ILU(0) breakdown：对角块 U_" +
                                        std::to_string(I) + std::to_string(I) + " 奇异");
        }

        // 4. 返回 apply (前代 + 回代)
        // 修复：因为 data 是 shared_ptr，按值捕获是完美且安全的
        return [data, nb](const std::vector<T>& rhs, std::vector<T>& z)
        {
            auto& y = data->y_buf;
            y.resize((size_t)nb * BS);
            z.resize((size_t)nb * BS);

            // 前代
            for (uint32_t I = 0; I < nb; ++I)
            {
                T* yI = &y[(size_t)I * BS];
                for (int c = 0; c < BS; ++c) yI[c] = rhs[(size_t)I * BS + c];
                for (uint32_t p = data->L_row_ptr[I]; p < data->L_row_ptr[I+1]; ++p)
                {
                    const T* LIJ = &data->L_values[(size_t)p * BV];
                    const T* yJ = &y[(size_t)data->L_col_idx[p] * BS];
                    for (int r = 0; r < BS; ++r)
                    {
                        T s = 0;
                        for (int c = 0; c < BS; ++c) s += LIJ[r*BS+c] * yJ[c];
                        yI[r] -= s;
                    }
                }
            }

            // 回代
            for (uint32_t II = nb; II-- > 0; )
            {
                T tmp[BS];
                for (int c = 0; c < BS; ++c) tmp[c] = y[(size_t)II * BS + c];
                for (uint32_t p = data->U_row_ptr[II]; p < data->U_row_ptr[II+1]; ++p)
                {
                    const uint32_t J = data->U_col_idx[p];
                    if (J == II) continue;
                    const T* UIJ = &data->U_values[(size_t)p * BV];
                    const T* zJ = &z[(size_t)J * BS];
                    for (int r = 0; r < BS; ++r)
                    {
                        T s = 0;
                        for (int c = 0; c < BS; ++c) s += UIJ[r*BS+c] * zJ[c];
                        tmp[r] -= s;
                    }
                }
                blk_matvec<BS>(&data->inv_diag[(size_t)II * BV], tmp, &z[(size_t)II * BS]);
            }
        };
    }

    // ---- 舒尔补算法 ----

    // 1. 鞍点系统结构体 (持有子矩阵的 spmv 接口)
    struct SaddlePointSystem {
        size_t n_vel;
        size_t n_pres;
        std::function<void(const std::vector<double>&, std::vector<double>&)> spmv_A;
        std::function<void(const std::vector<double>&, std::vector<double>&)> spmv_B;
        std::function<void(const std::vector<double>&, std::vector<double>&)> spmv_Bt;
    };

    // 2. Schur 补预处理工厂函数
    std::function<void(const std::vector<double>&, std::vector<double>&)> make_schur_preconditioner(
        const SaddlePointSystem& sys,
        std::function<void(const std::vector<double>&, std::vector<double>&)> A_precond,
        std::function<void(const std::vector<double>&, std::vector<double>&)> S_precond)
    {
        // 使用 mutable 允许在 lambda 内部修改捕获的缓冲区，避免每次 apply 重新分配内存
        return [sys, A_precond, S_precond](const std::vector<double>& r, std::vector<double>& z) mutable
        {
            const size_t n_vel = sys.n_vel;
            const size_t n_pres = sys.n_pres;
            
            // 预分配工作缓冲区 (首次调用时分配，后续复用)
            static thread_local std::vector<double> r_u, r_p, w_u, w_p, r_p_mod, z_p, w_bt, r_u_mod, z_u;
            r_u.resize(n_vel); r_p.resize(n_pres);
            w_u.resize(n_vel); w_p.resize(n_pres);
            r_p_mod.resize(n_pres); z_p.resize(n_pres);
            w_bt.resize(n_vel); r_u_mod.resize(n_vel); z_u.resize(n_vel);

            // 拆分 r
            for (size_t i = 0; i < n_vel; ++i) r_u[i] = r[i];
            for (size_t i = 0; i < n_pres; ++i) r_p[i] = r[n_vel + i];
            
            // 1. w_u = A^{-1} r_u
            A_precond(r_u, w_u);
            
            // 2. w_p = B * w_u
            sys.spmv_B(w_u, w_p);
            
            // 3. r_p_mod = w_p - r_p
            for (size_t i = 0; i < n_pres; ++i) r_p_mod[i] = w_p[i] - r_p[i];
            
            // 4. z_p = S^{-1} r_p_mod
            S_precond(r_p_mod, z_p);
            
            // 5. w_bt = B^T * z_p
            sys.spmv_Bt(z_p, w_bt);
            
            // 6. r_u_mod = r_u - w_bt
            for (size_t i = 0; i < n_vel; ++i) r_u_mod[i] = r_u[i] - w_bt[i];
            
            // 7. z_u = A^{-1} r_u_mod
            A_precond(r_u_mod, z_u);
            
            // 组装 z
            z.resize(n_vel + n_pres);
            for (size_t i = 0; i < n_vel; ++i) z[i] = z_u[i];
            for (size_t i = 0; i < n_pres; ++i) z[n_vel + i] = z_p[i];
        };
    }

    // Block 版本的 gmres
    template <typename T, int BS>
    std::tuple<std::vector<T>, int> BlockMatrix<T, BS>::gmres(
        const BlockMatrix& A,
        const std::vector<T>& b,
        const std::string& precond,
        double tol,
        int max_iter,
        int restart)
    {
        const size_t n = b.size();
        const int m = restart;
        std::vector<T> x(n, T(0));
        
        // 计算 ||b||
        T normb_sq = T(0);
        for (size_t i = 0; i < n; ++i) normb_sq += b[i] * b[i];
        const T normb = std::sqrt(normb_sq);
        if (normb < 1e-300) return {x, 0};

        // ====== 字符串分发 ======
        std::function<void(const std::vector<T>&, std::vector<T>&)> solveM;
        
        if (precond == "none" || precond == "identity")
        {
            solveM = [](const std::vector<T>& r, std::vector<T>& z) { z = r; };
        }
        else if (precond == "blockjacobi")
        {
            // 构造块 Jacobi 预处理器
            solveM = [A](const std::vector<T>& r, std::vector<T>& z) {
                z.resize(r.size());
                const uint32_t nb = A.block_rows();
                #ifdef CPPNUM_HAVE_OPENMP
                    #pragma omp parallel for schedule(static)
                #endif
                for (std::ptrdiff_t bi = 0; bi < (std::ptrdiff_t)nb; ++bi)
                {
                    const T* D = A.diag_block(bi);
                    // 这里需要调用块求逆，简化起见直接用高斯消元
                    // 实际应调用 blk_invert<BS>
                    T tmp[BS];
                    for (int c = 0; c < BS; ++c) tmp[c] = r[(size_t)bi * BS + c];
                    T Minv[BS * BS];
                    blk_invert<BS>(D, Minv);
                    blk_matvec<BS>(Minv, tmp, &z[(size_t)bi * BS]);
                }
            };
        }
        else if (precond == "blockilu" || precond == "blockilu0")
        {
            // 构造 Block-ILU(0) 预处理器
            solveM = make_block_ilu0_preconditioner(A);
        }
        else
        {
            throw std::invalid_argument("BlockMatrix::gmres 未知预处理类型: " + precond);
        }

        // ====== GMRES 核心循环（与 SMatrix::gmres 完全相同，只是 spmv 改为 A.matvec）======
        const T breakdown_tol = 1e-30;
        std::vector<std::vector<T>> V(m + 1, std::vector<T>(n, T(0)));
        std::vector<std::vector<T>> H(m + 1, std::vector<T>(m, T(0)));
        std::vector<T> cs(m, T(0)), sn(m, T(0));
        std::vector<T> g(m + 1, T(0));
        std::vector<T> y(m, T(0));
        std::vector<T> w(n), wtmp(n), z(n), dz(n);

        int total = 0;
        int it_end = 0;
        bool converged = false;

        while (total < max_iter && !converged)
        {
            //std::cout << "第" << total << "次迭代" << std::endl;
            T beta;
            if (total == 0)
            {
                beta = normb;
                for (size_t i = 0; i < n; ++i) V[0][i] = b[i] / beta;
            }
            else
            {
                std::vector<T> Ax;
                A.matvec(x, Ax);
                T rr = T(0);
                for (size_t i = 0; i < n; ++i)
                {
                    T ri = b[i] - Ax[i];
                    V[0][i] = ri;
                    rr += ri * ri;
                }
                beta = std::sqrt(rr);
                if (beta / normb < tol) { converged = true; break; }
                for (size_t i = 0; i < n; ++i) V[0][i] /= beta;
            }

            g.assign(m + 1, T(0));
            g[0] = beta;
            for (int i = 0; i < m + 1; ++i) H[i].assign(m, T(0));

            int j_used = 0;

            for (int j = 0; j < m && total < max_iter; ++j)
            {
                solveM(V[j], wtmp);
                A.matvec(wtmp, w);  // w = A * M^{-1} * v_j

                for (int i = 0; i <= j; ++i)
                {
                    T hij = T(0);
                    for (size_t t = 0; t < n; ++t) hij += w[t] * V[i][t];
                    H[i][j] = hij;
                    for (size_t t = 0; t < n; ++t) w[t] -= hij * V[i][t];
                }

                T wnorm_sq = T(0);
                for (size_t t = 0; t < n; ++t) wnorm_sq += w[t] * w[t];
                H[j + 1][j] = std::sqrt(wnorm_sq);
                
                if (H[j + 1][j] > breakdown_tol)
                {
                    const T inv = T(1) / H[j + 1][j];
                    for (size_t t = 0; t < n; ++t) V[j + 1][t] = w[t] * inv;
                }

                for (int i = 0; i < j; ++i)
                {
                    const T t1 =  cs[i] * H[i][j] + sn[i] * H[i + 1][j];
                    const T t2 = -sn[i] * H[i][j] + cs[i] * H[i + 1][j];
                    H[i][j] = t1;
                    H[i + 1][j] = t2;
                }
                const T den = std::hypot(H[j][j], H[j + 1][j]);
                if (den < breakdown_tol)
                    throw std::runtime_error("GMRES breakdown: Hessenberg pivot too small");
                cs[j] = H[j][j] / den;
                sn[j] = H[j + 1][j] / den;
                H[j][j] = cs[j] * H[j][j] + sn[j] * H[j + 1][j];
                H[j + 1][j] = T(0);
                const T g1 =  cs[j] * g[j] + sn[j] * g[j + 1];
                const T g2 = -sn[j] * g[j] + cs[j] * g[j + 1];
                g[j] = g1;
                g[j + 1] = g2;

                j_used = j + 1;
                it_end = total;
                ++total;

                if (std::fabs(g[j + 1]) / normb < tol)
                {
                    converged = true;
                    break;
                }
            }

            for (int i = j_used - 1; i >= 0; --i)
            {
                T sum = g[i];
                for (int k = i + 1; k < j_used; ++k)
                    sum -= H[i][k] * y[k];
                if (std::fabs(H[i][i]) < breakdown_tol)
                    throw std::runtime_error("GMRES breakdown: upper triangular pivot too small");
                y[i] = sum / H[i][i];
            }

            for (size_t t = 0; t < n; ++t)
            {
                T acc = T(0);
                for (int i = 0; i < j_used; ++i)
                    acc += y[i] * V[i][t];
                z[t] = acc;
            }
            solveM(z, dz);
            for (size_t t = 0; t < n; ++t)
                x[t] += dz[t];
        }

        return {x, it_end};
    }

    // =====================================================================
    // Block AMG (块代数多重网格) — 光滑聚合方法
    // 需要在文件顶部添加: #include <map>  #include <memory>
    // =====================================================================

    // ---- 辅助: 块 Frobenius 范数 ----
    template <typename T, int BS>
    inline T blk_fro_norm(const T* blk) 
    {
        T s = 0;
        for (int i = 0; i < BS * BS; ++i) s += blk[i] * blk[i];
        return std::sqrt(s);
    }

    // ---- 辅助: 块矩阵乘法 C = A * B (密集累加器) ----
    template <typename T, int BS>
    BlockMatrix<T, BS> blk_matmul(const BlockMatrix<T, BS>& A, const BlockMatrix<T, BS>& B) 
    {
        const int BV = BS * BS;
        const uint32_t nr = A.block_rows();
        const uint32_t nc = B.block_cols();

        std::vector<uint32_t> rp(nr + 1, 0);
        std::vector<uint32_t> ci;
        std::vector<T> val;

        std::vector<T> accum((size_t)nc * BV);
        std::vector<char> hit(nc, 0);
        std::vector<uint32_t> hit_list;

        for (uint32_t i = 0; i < nr; ++i) {
            std::fill(hit.begin(), hit.end(), 0);
            hit_list.clear();

            for (uint32_t pa = A.block_row_ptr()[i]; pa < A.block_row_ptr()[i+1]; ++pa) {
                uint32_t k = A.block_col_idx()[pa];
                const T* ab = &A.values()[(size_t)pa * BV];
                for (uint32_t pb = B.block_row_ptr()[k]; pb < B.block_row_ptr()[k+1]; ++pb) {
                    uint32_t j = B.block_col_idx()[pb];
                    const T* bb = &B.values()[(size_t)pb * BV];
                    if (!hit[j]) {
                        hit[j] = 1;
                        hit_list.push_back(j);
                        std::fill(accum.begin() + (size_t)j*BV,
                                accum.begin() + (size_t)(j+1)*BV, T(0));
                    }
                    T* cb = &accum[(size_t)j * BV];
                    for (int r = 0; r < BS; ++r)
                        for (int c = 0; c < BS; ++c) {
                            T s = 0;
                            for (int m = 0; m < BS; ++m) s += ab[r*BS+m] * bb[m*BS+c];
                            cb[r*BS+c] += s;
                        }
                }
            }
            std::sort(hit_list.begin(), hit_list.end());
            rp[i+1] = (uint32_t)hit_list.size();
            for (uint32_t j : hit_list) {
                ci.push_back(j);
                for (int v = 0; v < BV; ++v) val.push_back(accum[(size_t)j*BV + v]);
            }
        }
        for (uint32_t i = 0; i < nr; ++i) rp[i+1] += rp[i];
        return BlockMatrix<T, BS>(nr, nc, rp, ci, val);
    }

    // ---- 层级数据 ----
    template <typename T, int BS>
    struct BlockAMGLevel 
    {
        BlockMatrix<T, BS> A;      // 当前层算子
        BlockMatrix<T, BS> P;      // 延拓 (细→粗)
        BlockMatrix<T, BS> R;      // 限制 (粗→细), R = P^T
        std::vector<T> diag_inv;   // 对角块逆 (nrows × BS²)
        uint32_t nrows;
    };

    // ---- V-cycle 递归 ----
    template <typename T, int BS>
    void blk_amg_vcycle(const std::vector<BlockAMGLevel<T, BS>>& levels,
                        int lvl, int n_smooth, T omega,
                        const std::vector<T>& rhs, std::vector<T>& sol)
    {
        const int BV = BS * BS;
        const auto& L = levels[lvl];
        const uint32_t n    = L.nrows;
        const uint32_t ndof = n * BS;
        sol.resize(ndof);

        // ======== 最粗层: 稠密直接求解 ========
        if (lvl == (int)levels.size() - 1) {
            int sz = (int)ndof;
            std::vector<T> dense((size_t)sz * sz, T(0));
            for (uint32_t i = 0; i < n; ++i)
                for (uint32_t p = L.A.block_row_ptr()[i]; p < L.A.block_row_ptr()[i+1]; ++p) {
                    uint32_t j = L.A.block_col_idx()[p];
                    const T* blk = &L.A.values()[(size_t)p * BV];
                    for (int r = 0; r < BS; ++r)
                        for (int c = 0; c < BS; ++c)
                            dense[(size_t)(i*BS+r)*sz + j*BS+c] = blk[r*BS+c];
                }
            // 高斯消元 (部分主元)
            for (int k = 0; k < sz; ++k) {
                int pk = k;
                for (int i = k+1; i < sz; ++i)
                    if (std::fabs(dense[(size_t)i*sz+k]) > std::fabs(dense[(size_t)pk*sz+k])) pk = i;
                if (pk != k)
                    for (int j = 0; j < sz; ++j)
                        std::swap(dense[(size_t)k*sz+j], dense[(size_t)pk*sz+j]);
                T d = dense[(size_t)k*sz+k];
                if (std::fabs(d) < 1e-14) d = (d >= 0 ? 1e-14 : -1e-14);
                for (int i = k+1; i < sz; ++i) {
                    T f = dense[(size_t)i*sz+k] / d;
                    for (int j = k; j < sz; ++j)
                        dense[(size_t)i*sz+j] -= f * dense[(size_t)k*sz+j];
                }
            }
            std::vector<T> x(rhs.begin(), rhs.end());
            for (int k = 0; k < sz; ++k) {
                T d = dense[(size_t)k*sz+k];
                if (std::fabs(d) < 1e-14) d = (d >= 0 ? 1e-14 : -1e-14);
                for (int i = k+1; i < sz; ++i)
                    x[i] -= (dense[(size_t)i*sz+k] / d) * x[k];
            }
            for (int i = sz-1; i >= 0; --i) {
                for (int j = i+1; j < sz; ++j) x[i] -= dense[(size_t)i*sz+j] * x[j];
                T d = dense[(size_t)i*sz+i];
                if (std::fabs(d) < 1e-14) d = (d >= 0 ? 1e-14 : -1e-14);
                x[i] /= d;
            }
            sol = x;
            return;
        }

        // ======== 预光滑: Block Jacobi ========
        std::vector<T> tmp(ndof), r(ndof);
        for (int s = 0; s < n_smooth; ++s) {
            L.A.matvec(sol, tmp);
            for (uint32_t i = 0; i < ndof; ++i) r[i] = rhs[i] - tmp[i];
            for (uint32_t i = 0; i < n; ++i) {
                const T* dinv = &L.diag_inv[(size_t)i * BV];
                for (int c = 0; c < BS; ++c) {
                    T corr = 0;
                    for (int rr = 0; rr < BS; ++rr)
                        corr += dinv[c*BS+rr] * r[i*BS+rr];
                    sol[i*BS+c] += omega * corr;
                }
            }
        }

        // ======== 残差 → 限制 → 递归 → 延拓修正 ========
        L.A.matvec(sol, tmp);
        for (uint32_t i = 0; i < ndof; ++i) r[i] = rhs[i] - tmp[i];

        std::vector<T> rc;
        L.R.matvec(r, rc);                    // r_c = R r

        std::vector<T> ec;
        blk_amg_vcycle(levels, lvl+1, n_smooth, omega, rc, ec);  // 递归

        std::vector<T> pec;
        L.P.matvec(ec, pec);                  // P e_c
        for (uint32_t i = 0; i < ndof; ++i) sol[i] += pec[i];

        // ======== 后光滑: Block Jacobi ========
        for (int s = 0; s < n_smooth; ++s) {
            L.A.matvec(sol, tmp);
            for (uint32_t i = 0; i < ndof; ++i) r[i] = rhs[i] - tmp[i];
            for (uint32_t i = 0; i < n; ++i) {
                const T* dinv = &L.diag_inv[(size_t)i * BV];
                for (int c = 0; c < BS; ++c) {
                    T corr = 0;
                    for (int rr = 0; rr < BS; ++rr)
                        corr += dinv[c*BS+rr] * r[i*BS+rr];
                    sol[i*BS+c] += omega * corr;
                }
            }
        }
    }

    // =====================================================================
    // 主函数: 创建 Block AMG 预处理器
    // =====================================================================
    template <typename T, int BS>
    std::function<void(const std::vector<T>&, std::vector<T>&)>
    make_block_amg_preconditioner(const BlockMatrix<T, BS>& A)
    {
        if (A.block_rows() != A.block_cols())
            throw std::invalid_argument("Block AMG 要求方阵");

        const int BV = BS * BS;
        const T theta        = 0.25;            // 强度连接阈值
        const T omega_smooth = T(2) / T(3);     // Jacobi 阻尼
        const int n_smooth   = 1;               // 前/后光滑步数
        const int max_levels = 20;              // 最大层数
        const int max_agg    = 4;               // 最大聚合大小 (块行数)
        const uint32_t coarse_max = 50;         // 最粗层最大块行数

        auto lv_ptr = std::make_shared<std::vector<BlockAMGLevel<T, BS>>>();
        auto& levels = *lv_ptr;
        BlockMatrix<T, BS> cur_A = A;

        for (int lvl = 0; lvl < max_levels; ++lvl) {
            uint32_t n = cur_A.block_rows();

            BlockAMGLevel<T, BS> level;
            level.A     = cur_A;
            level.nrows = n;

            // 对角块逆
            level.diag_inv.resize((size_t)n * BV);
            for (uint32_t i = 0; i < n; ++i)
                blk_invert<BS>(cur_A.diag_block(i), &level.diag_inv[(size_t)i * BV]);

            if (n <= coarse_max) { levels.push_back(std::move(level)); break; }

            // ---- 1. 强度连接 (Frobenius 范数) ----
            std::vector<std::vector<uint32_t>> strong(n);
            for (uint32_t i = 0; i < n; ++i) {
                T mx = 0;
                for (uint32_t p = cur_A.block_row_ptr()[i]; p < cur_A.block_row_ptr()[i+1]; ++p) {
                    uint32_t j = cur_A.block_col_idx()[p];
                    if (j != i)
                        mx = std::max(mx, blk_fro_norm<T, BS>(&cur_A.values()[(size_t)p*BV]));
                }
                T thr = theta * mx;
                for (uint32_t p = cur_A.block_row_ptr()[i]; p < cur_A.block_row_ptr()[i+1]; ++p) {
                    uint32_t j = cur_A.block_col_idx()[p];
                    if (j != i && blk_fro_norm<T, BS>(&cur_A.values()[(size_t)p*BV]) >= thr)
                        strong[i].push_back(j);
                }
            }

            // ---- 2. 贪心聚合 ----
            std::vector<int> agg(n, -1);
            int n_agg = 0;
            for (uint32_t i = 0; i < n; ++i) {
                if (agg[i] != -1) continue;
                agg[i] = n_agg;
                int sz = 1;
                for (uint32_t j : strong[i])
                    if (agg[j] == -1 && sz < max_agg) { agg[j] = n_agg; ++sz; }
                ++n_agg;
            }

            // ---- 3. 分片常数延拓 P_t:  P_t[i][agg[i]] = I ----
            std::vector<uint32_t> pt_rp(n + 1);
            std::vector<uint32_t> pt_ci(n);
            std::vector<T> pt_val((size_t)n * BV, T(0));
            for (uint32_t i = 0; i < n; ++i) {
                pt_rp[i] = i;
                pt_ci[i] = (uint32_t)agg[i];
                for (int r = 0; r < BS; ++r) pt_val[(size_t)i*BV + r*BS + r] = T(1);
            }
            pt_rp[n] = n;
            BlockMatrix<T, BS> Pt(n, (uint32_t)n_agg, pt_rp, pt_ci, pt_val);

            // ---- 4. 光滑延拓 P = (I − ω D⁻¹ A) P_t ----
            std::vector<std::vector<std::pair<uint32_t, std::vector<T>>>> p_rows(n);
            for (uint32_t i = 0; i < n; ++i) {
                std::map<uint32_t, std::vector<T>> row_blks;

                // 恒等项
                row_blks[(uint32_t)agg[i]] = std::vector<T>(BV, T(0));
                for (int r = 0; r < BS; ++r)
                    row_blks[(uint32_t)agg[i]][r*BS+r] = T(1);

                // 光滑项: −ω D_i⁻¹ A[i][k]
                const T* dinv = &level.diag_inv[(size_t)i * BV];
                for (uint32_t p = cur_A.block_row_ptr()[i]; p < cur_A.block_row_ptr()[i+1]; ++p) {
                    uint32_t k = cur_A.block_col_idx()[p];
                    uint32_t g = (uint32_t)agg[k];
                    const T* ablk = &cur_A.values()[(size_t)p * BV];
                    if (!row_blks.count(g)) row_blks[g] = std::vector<T>(BV, T(0));
                    for (int r = 0; r < BS; ++r)
                        for (int c = 0; c < BS; ++c) {
                            T s = 0;
                            for (int m = 0; m < BS; ++m) s += dinv[r*BS+m] * ablk[m*BS+c];
                            row_blks[g][r*BS+c] -= omega_smooth * s;
                        }
                }
                for (auto& kv : row_blks) p_rows[i].push_back(kv);
            }

            std::vector<uint32_t> p_rp(n + 1, 0);
            std::vector<uint32_t> p_ci;
            std::vector<T> p_val;
            for (uint32_t i = 0; i < n; ++i) {
                p_rp[i+1] = (uint32_t)p_rows[i].size();
                for (auto& kv : p_rows[i]) {
                    p_ci.push_back(kv.first);
                    for (int v = 0; v < BV; ++v) p_val.push_back(kv.second[v]);
                }
            }
            for (uint32_t i = 0; i < n; ++i) p_rp[i+1] += p_rp[i];

            level.P = BlockMatrix<T, BS>(n, (uint32_t)n_agg, p_rp, p_ci, p_val);
            level.R = level.P.transpose();
            levels.push_back(std::move(level));

            // ---- 5. Galerkin: A_c = P^T A P ----
            BlockMatrix<T, BS> AP = blk_matmul(cur_A, levels.back().P);
            cur_A = blk_matmul(levels.back().R, AP);
        }

        // ---- 返回 V-cycle lambda ----
        return [lv_ptr, n_smooth, omega_smooth](const std::vector<T>& rhs,
                                                std::vector<T>& sol) {
            sol.assign(rhs.size(), T(0));
            blk_amg_vcycle(*lv_ptr, 0, n_smooth, omega_smooth, rhs, sol);
        };
    }
}