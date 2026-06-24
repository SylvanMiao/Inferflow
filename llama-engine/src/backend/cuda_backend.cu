#include "llama/backend/cuda_backend.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <stdexcept>
#include <string>
#include <vector>

namespace llama {
namespace backend {
namespace {

void check_cuda(cudaError_t status, const char* what) {
    if (status != cudaSuccess) {
        throw std::runtime_error(std::string(what) + ": " + cudaGetErrorString(status));
    }
}

template <typename T>
class DeviceBuffer {
public:
    explicit DeviceBuffer(size_t count) : count_(count) {
        if (count_ > 0) {
            check_cuda(cudaMalloc(&ptr_, count_ * sizeof(T)), "cudaMalloc");
        }
    }

    ~DeviceBuffer() {
        if (ptr_) cudaFree(ptr_);
    }

    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;

    DeviceBuffer(DeviceBuffer&& other) noexcept : ptr_(other.ptr_), count_(other.count_) {
        other.ptr_ = nullptr;
        other.count_ = 0;
    }

    DeviceBuffer& operator=(DeviceBuffer&& other) noexcept {
        if (this != &other) {
            if (ptr_) cudaFree(ptr_);
            ptr_ = other.ptr_;
            count_ = other.count_;
            other.ptr_ = nullptr;
            other.count_ = 0;
        }
        return *this;
    }

    T* get() { return ptr_; }
    const T* get() const { return ptr_; }
    size_t count() const { return count_; }

    void copy_from_host(const T* src, size_t count) {
        if (count > count_) throw std::runtime_error("copy_from_host count overflow");
        check_cuda(cudaMemcpy(ptr_, src, count * sizeof(T), cudaMemcpyHostToDevice),
                   "cudaMemcpy H2D");
    }

    void copy_to_host(T* dst, size_t count) const {
        if (count > count_) throw std::runtime_error("copy_to_host count overflow");
        check_cuda(cudaMemcpy(dst, ptr_, count * sizeof(T), cudaMemcpyDeviceToHost),
                   "cudaMemcpy D2H");
    }

private:
    T* ptr_ = nullptr;
    size_t count_ = 0;
};

__global__ void rmsnorm_kernel(const float* input,
                               const float* weight,
                               float* output,
                               int size,
                               float eps) {
    __shared__ float partial[256];
    const int tid = threadIdx.x;

    float sum = 0.0f;
    for (int i = tid; i < size; i += blockDim.x) {
        sum += input[i] * input[i];
    }
    partial[tid] = sum;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            partial[tid] += partial[tid + stride];
        }
        __syncthreads();
    }

    const float scale = rsqrtf(partial[0] / static_cast<float>(size) + eps);
    for (int i = tid; i < size; i += blockDim.x) {
        output[i] = input[i] * weight[i] * scale;
    }
}

__global__ void matmul_kernel(const float* input,
                              const float* weight,
                              float* output,
                              int rows,
                              int cols) {
    __shared__ float partial[256];
    const int row = blockIdx.x;
    const int tid = threadIdx.x;
    if (row >= rows) return;

    float sum = 0.0f;
    const int offset = row * cols;
    for (int c = tid; c < cols; c += blockDim.x) {
        sum += input[c] * weight[offset + c];
    }
    partial[tid] = sum;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) partial[tid] += partial[tid + stride];
        __syncthreads();
    }
    if (tid == 0) output[row] = partial[0];
}

__global__ void matmul_int8_kernel(const float* input,
                                   const int8_t* weight,
                                   const float* scales,
                                   float* output,
                                   int rows,
                                   int cols,
                                   int group_size) {
    __shared__ float partial[256];
    const int row = blockIdx.x;
    const int tid = threadIdx.x;
    if (row >= rows) return;

    float sum = 0.0f;
    const int offset = row * cols;
    for (int c = tid; c < cols; c += blockDim.x) {
        const int idx = offset + c;
        const int group = idx / group_size;
        sum += input[c] * static_cast<float>(weight[idx]) * scales[group];
    }
    partial[tid] = sum;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) partial[tid] += partial[tid + stride];
        __syncthreads();
    }
    if (tid == 0) output[row] = partial[0];
}

__global__ void rope_kernel(float* q,
                            float* k,
                            int q_rows,
                            int k_rows,
                            int head_dim,
                            const float* cos_cache,
                            const float* sin_cache,
                            int cache_cols,
                            int pos) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int half_dim = head_dim / 2;
    const int q_total = q_rows * half_dim;
    const int k_total = k_rows * half_dim;
    const int total = q_total + k_total;
    if (idx >= total) return;

    float* base = q;
    int local = idx;
    int rows = q_rows;
    if (idx >= q_total) {
        base = k;
        local = idx - q_total;
        rows = k_rows;
    }

    const int row = local / half_dim;
    const int col = local % half_dim;
    if (row >= rows) return;

    const float c = cos_cache[pos * cache_cols + col];
    const float s = sin_cache[pos * cache_cols + col];
    const int offset = row * head_dim;
    const float x0 = base[offset + col];
    const float x1 = base[offset + col + half_dim];
    base[offset + col] = x0 * c - x1 * s;
    base[offset + col + half_dim] = x0 * s + x1 * c;
}

__global__ void rope_row_kernel(float* q,
                                float* k,
                                int q_rows,
                                int k_rows,
                                int head_dim,
                                const float* cos_row,
                                const float* sin_row) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int half_dim = head_dim / 2;
    const int q_total = q_rows * half_dim;
    const int k_total = k_rows * half_dim;
    const int total = q_total + k_total;
    if (idx >= total) return;

    float* base = q;
    int local = idx;
    int rows = q_rows;
    if (idx >= q_total) {
        base = k;
        local = idx - q_total;
        rows = k_rows;
    }

    const int row = local / half_dim;
    const int col = local % half_dim;
    if (row >= rows) return;

    const float c = cos_row[col];
    const float s = sin_row[col];
    const int offset = row * head_dim;
    const float x0 = base[offset + col];
    const float x1 = base[offset + col + half_dim];
    base[offset + col] = x0 * c - x1 * s;
    base[offset + col + half_dim] = x0 * s + x1 * c;
}

using RowMatrix = Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;

// ═══════════════════════════════════════════════════════════════════════════
// Attention kernel — multi-head attention with GQA for single-token decode
// ═══════════════════════════════════════════════════════════════════════════
//
// One block per query head.  Each block:
//   1. Loads q[h] into shared memory
//   2. Computes dot(q[h], K[head_kv(h), t]) * scale for every position t
//   3. Softmax over positions (two-pass with shared-mem reduction)
//   4. Weighted sum of V[head_kv(h), t] → output[h]
//
// K and V are expected in flattened row-major layout:
//   [n_kv_heads, cache_stride]  (each head's data is cache_stride floats,
//                                first seq_len*head_dim entries are valid)

__global__ void attention_kernel(const float* q,
                                  const float* k_cache,
                                  const float* v_cache,
                                  float* output,
                                  int seq_len,
                                  int cache_stride,
                                  int head_dim,
                                  int n_heads,
                                  int n_kv_heads) {
    extern __shared__ float shared[];
    // shared layout: scores[seq_len] then q_shared[head_dim] then workspace[blockDim.x]
    float* scores    = shared;
    float* q_shared  = shared + seq_len;
    float* workspace = shared + seq_len + head_dim;

    const int h = blockIdx.x;
    if (h >= n_heads) return;

    const int n_groups = n_heads / n_kv_heads;
    const int kv_h = h / n_groups;
    const int tid = threadIdx.x;

    // ---- Phase 1: load q[h] into shared memory ----
    for (int i = tid; i < head_dim; i += blockDim.x) {
        q_shared[i] = q[h * head_dim + i];
    }
    __syncthreads();

    const float scale = rsqrtf(static_cast<float>(head_dim));
    const int kv_offset = kv_h * cache_stride;

    // ---- Phase 2: compute dot(q[h], K[t]) * scale for each position ----
    float local_max = -1e30f;
    for (int t = tid; t < seq_len; t += blockDim.x) {
        float dot = 0.0f;
        for (int d = 0; d < head_dim; ++d) {
            dot += q_shared[d] * k_cache[kv_offset + t * head_dim + d];
        }
        scores[t] = dot * scale;
        local_max = fmaxf(local_max, scores[t]);
    }
    __syncthreads();

    // ---- Phase 3: block-reduce to find global max ----
    workspace[tid] = local_max;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            workspace[tid] = fmaxf(workspace[tid], workspace[tid + stride]);
        }
        __syncthreads();
    }
    const float max_score = workspace[0];
    __syncthreads();

    // ---- Phase 4: exp(scores - max) and reduce sum ----
    float local_sum = 0.0f;
    for (int t = tid; t < seq_len; t += blockDim.x) {
        scores[t] = expf(scores[t] - max_score);
        local_sum += scores[t];
    }
    workspace[tid] = local_sum;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            workspace[tid] += workspace[tid + stride];
        }
        __syncthreads();
    }
    const float inv_sum = 1.0f / workspace[0];
    __syncthreads();

    // ---- Phase 5: normalize scores ----
    for (int t = tid; t < seq_len; t += blockDim.x) {
        scores[t] *= inv_sum;
    }
    __syncthreads();

    // ---- Phase 6: weighted sum of V ----
    const int v_offset = kv_h * cache_stride;
    for (int d = tid; d < head_dim; d += blockDim.x) {
        float sum = 0.0f;
        for (int t = 0; t < seq_len; ++t) {
            sum += scores[t] * v_cache[v_offset + t * head_dim + d];
        }
        output[h * head_dim + d] = sum;
    }
}

}  // namespace

struct CudaMatrix {
    int rows = 0;
    int cols = 0;
    DeviceBuffer<float> values;

    explicit CudaMatrix(const Eigen::MatrixXf& weight)
        : rows(static_cast<int>(weight.rows()))
        , cols(static_cast<int>(weight.cols()))
        , values(static_cast<size_t>(weight.rows()) * weight.cols())
    {
        RowMatrix row_weight = weight;
        values.copy_from_host(row_weight.data(), static_cast<size_t>(rows) * cols);
    }
};

bool cuda_available() {
    int count = 0;
    return cudaGetDeviceCount(&count) == cudaSuccess && count > 0;
}

void cuda_rmsnorm(const Eigen::VectorXf& input,
                  const Eigen::VectorXf& weight,
                  float eps,
                  Eigen::VectorXf& output) {
    if (input.size() != weight.size()) {
        throw std::invalid_argument("cuda_rmsnorm input/weight size mismatch");
    }
    const int size = static_cast<int>(input.size());
    output.resize(size);

    DeviceBuffer<float> d_input(size);
    DeviceBuffer<float> d_weight(size);
    DeviceBuffer<float> d_output(size);
    d_input.copy_from_host(input.data(), size);
    d_weight.copy_from_host(weight.data(), size);

    rmsnorm_kernel<<<1, 256>>>(d_input.get(), d_weight.get(), d_output.get(), size, eps);
    check_cuda(cudaGetLastError(), "rmsnorm_kernel launch");
    check_cuda(cudaDeviceSynchronize(), "rmsnorm_kernel sync");
    d_output.copy_to_host(output.data(), size);
}

void cuda_matmul(const Eigen::VectorXf& input,
                 const Eigen::MatrixXf& weight,
                 Eigen::VectorXf& output) {
    if (input.size() != weight.cols()) {
        throw std::invalid_argument("cuda_matmul input size does not match weight cols");
    }
    const int rows = static_cast<int>(weight.rows());
    const int cols = static_cast<int>(weight.cols());
    output.resize(rows);

    RowMatrix row_weight = weight;
    DeviceBuffer<float> d_input(cols);
    DeviceBuffer<float> d_weight(rows * cols);
    DeviceBuffer<float> d_output(rows);
    d_input.copy_from_host(input.data(), cols);
    d_weight.copy_from_host(row_weight.data(), rows * cols);

    matmul_kernel<<<rows, 256>>>(d_input.get(), d_weight.get(), d_output.get(), rows, cols);
    check_cuda(cudaGetLastError(), "matmul_kernel launch");
    check_cuda(cudaDeviceSynchronize(), "matmul_kernel sync");
    d_output.copy_to_host(output.data(), rows);
}

CudaMatrix* cuda_matrix_create(const Eigen::MatrixXf& weight) {
    return new CudaMatrix(weight);
}

void cuda_matrix_destroy(CudaMatrix* matrix) {
    delete matrix;
}

int cuda_matrix_rows(const CudaMatrix* matrix) {
    return matrix ? matrix->rows : 0;
}

int cuda_matrix_cols(const CudaMatrix* matrix) {
    return matrix ? matrix->cols : 0;
}

void cuda_matmul_device_weight(const Eigen::VectorXf& input,
                               const CudaMatrix* weight,
                               Eigen::VectorXf& output) {
    if (weight == nullptr) {
        throw std::invalid_argument("cuda_matmul_device_weight got null weight");
    }
    if (input.size() != weight->cols) {
        throw std::invalid_argument("cuda_matmul_device_weight input size does not match weight cols");
    }

    output.resize(weight->rows);
    DeviceBuffer<float> d_input(weight->cols);
    DeviceBuffer<float> d_output(weight->rows);
    d_input.copy_from_host(input.data(), weight->cols);

    matmul_kernel<<<weight->rows, 256>>>(d_input.get(), weight->values.get(), d_output.get(),
                                         weight->rows, weight->cols);
    check_cuda(cudaGetLastError(), "matmul_device_weight kernel launch");
    check_cuda(cudaDeviceSynchronize(), "matmul_device_weight kernel sync");
    d_output.copy_to_host(output.data(), weight->rows);
}

void cuda_matmul_int8(const Eigen::VectorXf& input,
                      const quant::QuantizedMatrixInt8& weight,
                      Eigen::VectorXf& output) {
    const std::string error = weight.validate();
    if (!error.empty()) {
        throw std::invalid_argument("invalid QuantizedMatrixInt8: " + error);
    }
    if (input.size() != weight.cols) {
        throw std::invalid_argument("cuda_matmul_int8 input size does not match weight cols");
    }
    output.resize(weight.rows);

    DeviceBuffer<float> d_input(weight.cols);
    DeviceBuffer<int8_t> d_weight(weight.values.size());
    DeviceBuffer<float> d_scales(weight.scales.size());
    DeviceBuffer<float> d_output(weight.rows);
    d_input.copy_from_host(input.data(), weight.cols);
    d_weight.copy_from_host(weight.values.data(), weight.values.size());
    d_scales.copy_from_host(weight.scales.data(), weight.scales.size());

    matmul_int8_kernel<<<weight.rows, 256>>>(d_input.get(), d_weight.get(), d_scales.get(),
                                            d_output.get(), weight.rows, weight.cols,
                                            weight.group_size);
    check_cuda(cudaGetLastError(), "matmul_int8_kernel launch");
    check_cuda(cudaDeviceSynchronize(), "matmul_int8_kernel sync");
    d_output.copy_to_host(output.data(), weight.rows);
}

void cuda_rope(Eigen::MatrixXf& q,
               Eigen::MatrixXf& k,
               int pos,
               const Eigen::MatrixXf& cos_cache,
               const Eigen::MatrixXf& sin_cache) {
    if (q.cols() != k.cols()) {
        throw std::invalid_argument("cuda_rope q/k head_dim mismatch");
    }
    if (pos < 0 || pos >= cos_cache.rows() || pos >= sin_cache.rows()) {
        throw std::invalid_argument("cuda_rope pos out of range");
    }
    if (cos_cache.cols() != q.cols() / 2 || sin_cache.cols() != q.cols() / 2) {
        throw std::invalid_argument("cuda_rope cache shape mismatch");
    }

    RowMatrix q_row = q;
    RowMatrix k_row = k;
    RowMatrix cos_row = cos_cache;
    RowMatrix sin_row = sin_cache;
    const int q_size = static_cast<int>(q_row.size());
    const int k_size = static_cast<int>(k_row.size());
    const int cache_size = static_cast<int>(cos_row.size());

    DeviceBuffer<float> d_q(q_size);
    DeviceBuffer<float> d_k(k_size);
    DeviceBuffer<float> d_cos(cache_size);
    DeviceBuffer<float> d_sin(cache_size);
    d_q.copy_from_host(q_row.data(), q_size);
    d_k.copy_from_host(k_row.data(), k_size);
    d_cos.copy_from_host(cos_row.data(), cache_size);
    d_sin.copy_from_host(sin_row.data(), cache_size);

    const int half_dim = static_cast<int>(q.cols()) / 2;
    const int work = static_cast<int>(q.rows() + k.rows()) * half_dim;
    const int threads = 128;
    const int blocks = (work + threads - 1) / threads;
    rope_kernel<<<blocks, threads>>>(d_q.get(), d_k.get(),
                                     static_cast<int>(q.rows()), static_cast<int>(k.rows()),
                                     static_cast<int>(q.cols()),
                                     d_cos.get(), d_sin.get(),
                                     static_cast<int>(cos_cache.cols()), pos);
    check_cuda(cudaGetLastError(), "rope_kernel launch");
    check_cuda(cudaDeviceSynchronize(), "rope_kernel sync");

    d_q.copy_to_host(q_row.data(), q_size);
    d_k.copy_to_host(k_row.data(), k_size);
    q = q_row;
    k = k_row;
}

void cuda_attention(const Eigen::VectorXf& q,
                     const Eigen::MatrixXf& k_cached,
                     const Eigen::MatrixXf& v_cached,
                     int seq_len,
                     int n_heads,
                     int n_kv_heads,
                     int head_dim,
                     Eigen::VectorXf& output) {
    if (seq_len <= 0) {
        throw std::invalid_argument("cuda_attention seq_len must be positive");
    }
    if (q.size() != n_heads * head_dim) {
        throw std::invalid_argument("cuda_attention q size mismatch");
    }
    if (k_cached.rows() < seq_len || k_cached.cols() != n_kv_heads * head_dim) {
        throw std::invalid_argument("cuda_attention k_cached shape mismatch");
    }
    if (v_cached.rows() < seq_len || v_cached.cols() != n_kv_heads * head_dim) {
        throw std::invalid_argument("cuda_attention v_cached shape mismatch");
    }

    output.resize(n_heads * head_dim);

    const int max_seq_len = static_cast<int>(k_cached.rows());

    // Flatten K and V from Eigen column-major [max_seq_len, kv_dim]
    // to GPU-friendly row-major [n_kv_heads, seq_len * head_dim].
    const size_t flat_size = static_cast<size_t>(n_kv_heads) * seq_len * head_dim;
    std::vector<float> k_flat(flat_size);
    std::vector<float> v_flat(flat_size);

    for (int kv_h = 0; kv_h < n_kv_heads; ++kv_h) {
        for (int t = 0; t < seq_len; ++t) {
            for (int d = 0; d < head_dim; ++d) {
                // k_cached is col-major: element (t, kv_h*head_dim + d)
                //   → flat offset = t + (kv_h*head_dim + d) * max_seq_len
                const int src_col = kv_h * head_dim + d;
                const float kval = k_cached(t, src_col);
                const float vval = v_cached(t, src_col);
                // k_flat layout: [n_kv_heads, seq_len * head_dim] row-major
                const size_t dst = static_cast<size_t>(kv_h) * seq_len * head_dim
                                 + static_cast<size_t>(t) * head_dim + d;
                k_flat[dst] = kval;
                v_flat[dst] = vval;
            }
        }
    }

    const size_t q_count = static_cast<size_t>(n_heads) * head_dim;

    DeviceBuffer<float> d_q(q_count);
    DeviceBuffer<float> d_k(flat_size);
    DeviceBuffer<float> d_v(flat_size);
    DeviceBuffer<float> d_out(q_count);

    d_q.copy_from_host(q.data(), q_count);
    d_k.copy_from_host(k_flat.data(), flat_size);
    d_v.copy_from_host(v_flat.data(), flat_size);

    const int threads = 256;
    const size_t shared_bytes = (static_cast<size_t>(seq_len) + head_dim + threads) * sizeof(float);
    attention_kernel<<<n_heads, threads, shared_bytes>>>(
        d_q.get(), d_k.get(), d_v.get(), d_out.get(),
        seq_len, seq_len * head_dim, head_dim, n_heads, n_kv_heads);
    check_cuda(cudaGetLastError(), "attention_kernel launch");
    check_cuda(cudaDeviceSynchronize(), "attention_kernel sync");

    d_out.copy_to_host(output.data(), q_count);
}

// ═══════════════════════════════════════════════════════════════════════════
// GPU-resident forward context — owns KV cache and intermediate buffers
// ═══════════════════════════════════════════════════════════════════════════

struct CudaForwardContext {
    int n_layers;
    int dim;
    int hidden_dim;
    int vocab_size;
    int kv_dim;
    int n_kv_heads;
    int head_dim;
    int max_seq_len;

    // GPU KV cache: per layer, layout [n_kv_heads, max_seq_len * head_dim]
    // Element (kv_h, pos * head_dim + d) at offset:
    //   kv_h * max_seq_len * head_dim + pos * head_dim + d
    std::vector<DeviceBuffer<float>> k_cache;
    std::vector<DeviceBuffer<float>> v_cache;
    DeviceBuffer<float> scratch_input;
    DeviceBuffer<float> scratch_weight;
    DeviceBuffer<float> scratch_output;
    DeviceBuffer<float> scratch_q;
    DeviceBuffer<float> scratch_k;
    DeviceBuffer<float> scratch_cos;
    DeviceBuffer<float> scratch_sin;
    DeviceBuffer<float> scratch_attn;

    CudaForwardContext(const ModelConfig& cfg)
        : n_layers(cfg.n_layers)
        , dim(cfg.dim)
        , hidden_dim(cfg.hidden_dim)
        , vocab_size(cfg.vocab_size)
        , kv_dim(cfg.kv_dim)
        , n_kv_heads(cfg.n_kv_heads)
        , head_dim(cfg.head_dim)
        , max_seq_len(cfg.max_seq_len)
        , scratch_input(std::max({cfg.dim, cfg.hidden_dim, cfg.vocab_size, cfg.kv_dim}))
        , scratch_weight(std::max({cfg.dim, cfg.hidden_dim, cfg.vocab_size, cfg.kv_dim}))
        , scratch_output(std::max({cfg.dim, cfg.hidden_dim, cfg.vocab_size, cfg.kv_dim}))
        , scratch_q(cfg.dim)
        , scratch_k(cfg.kv_dim)
        , scratch_cos(cfg.head_dim / 2)
        , scratch_sin(cfg.head_dim / 2)
        , scratch_attn(cfg.dim)
    {
        const size_t per_layer = static_cast<size_t>(n_kv_heads) * max_seq_len * head_dim;
        for (int i = 0; i < n_layers; ++i) {
            k_cache.emplace_back(per_layer);
            v_cache.emplace_back(per_layer);
        }
    }

    void reset() {
        for (int i = 0; i < n_layers; ++i) {
            check_cuda(cudaMemset(k_cache[i].get(), 0,
                                  k_cache[i].count() * sizeof(float)),
                       "kv cache reset k");
            check_cuda(cudaMemset(v_cache[i].get(), 0,
                                  v_cache[i].count() * sizeof(float)),
                       "kv cache reset v");
        }
    }

    /** Write k_data (CPU, [n_kv_heads * head_dim]) to GPU KV cache at pos. */
    void write_kv(int layer, int pos, const float* k_data, const float* v_data) {
        for (int h = 0; h < n_kv_heads; ++h) {
            const size_t src_off = static_cast<size_t>(h) * head_dim;
            const size_t dst_off =
                (static_cast<size_t>(h) * max_seq_len + pos) * head_dim;

            check_cuda(cudaMemcpy(k_cache[layer].get() + dst_off,
                                  k_data + src_off,
                                  head_dim * sizeof(float),
                                  cudaMemcpyHostToDevice),
                       "kv cache write k");
            check_cuda(cudaMemcpy(v_cache[layer].get() + dst_off,
                                  v_data + src_off,
                                  head_dim * sizeof(float),
                                  cudaMemcpyHostToDevice),
                       "kv cache write v");
        }
    }

    const float* k(int layer) const { return k_cache[layer].get(); }
    const float* v(int layer) const { return v_cache[layer].get(); }
};

CudaForwardContext* cuda_context_create(const ModelConfig& cfg) {
    return new CudaForwardContext(cfg);
}

void cuda_context_destroy(CudaForwardContext* ctx) {
    delete ctx;
}

void cuda_context_reset(CudaForwardContext* ctx) {
    if (ctx) ctx->reset();
}

void cuda_rmsnorm_cached(CudaForwardContext* ctx,
                         const Eigen::VectorXf& input,
                         const Eigen::VectorXf& weight,
                         float eps,
                         Eigen::VectorXf& output) {
    if (ctx == nullptr) {
        throw std::invalid_argument("cuda_rmsnorm_cached got null context");
    }
    if (input.size() != weight.size()) {
        throw std::invalid_argument("cuda_rmsnorm_cached input/weight size mismatch");
    }

    const int size = static_cast<int>(input.size());
    output.resize(size);
    ctx->scratch_input.copy_from_host(input.data(), size);
    ctx->scratch_weight.copy_from_host(weight.data(), size);

    rmsnorm_kernel<<<1, 256>>>(ctx->scratch_input.get(), ctx->scratch_weight.get(),
                               ctx->scratch_output.get(), size, eps);
    check_cuda(cudaGetLastError(), "rmsnorm_cached kernel launch");
    check_cuda(cudaDeviceSynchronize(), "rmsnorm_cached kernel sync");
    ctx->scratch_output.copy_to_host(output.data(), size);
}

void cuda_matmul_device_weight_cached(CudaForwardContext* ctx,
                                      const Eigen::VectorXf& input,
                                      const CudaMatrix* weight,
                                      Eigen::VectorXf& output) {
    if (ctx == nullptr) {
        throw std::invalid_argument("cuda_matmul_device_weight_cached got null context");
    }
    if (weight == nullptr) {
        throw std::invalid_argument("cuda_matmul_device_weight_cached got null weight");
    }
    if (input.size() != weight->cols) {
        throw std::invalid_argument("cuda_matmul_device_weight_cached input size mismatch");
    }

    output.resize(weight->rows);
    ctx->scratch_input.copy_from_host(input.data(), weight->cols);

    matmul_kernel<<<weight->rows, 256>>>(ctx->scratch_input.get(), weight->values.get(),
                                         ctx->scratch_output.get(), weight->rows, weight->cols);
    check_cuda(cudaGetLastError(), "matmul_device_weight_cached kernel launch");
    check_cuda(cudaDeviceSynchronize(), "matmul_device_weight_cached kernel sync");
    ctx->scratch_output.copy_to_host(output.data(), weight->rows);
}

void cuda_rope_cached(CudaForwardContext* ctx,
                      Eigen::MatrixXf& q,
                      Eigen::MatrixXf& k,
                      int pos,
                      const Eigen::MatrixXf& cos_cache,
                      const Eigen::MatrixXf& sin_cache) {
    if (ctx == nullptr) {
        throw std::invalid_argument("cuda_rope_cached got null context");
    }
    if (q.cols() != k.cols()) {
        throw std::invalid_argument("cuda_rope_cached q/k head_dim mismatch");
    }
    if (pos < 0 || pos >= cos_cache.rows() || pos >= sin_cache.rows()) {
        throw std::invalid_argument("cuda_rope_cached pos out of range");
    }
    if (cos_cache.cols() != q.cols() / 2 || sin_cache.cols() != q.cols() / 2) {
        throw std::invalid_argument("cuda_rope_cached cache shape mismatch");
    }

    RowMatrix q_row = q;
    RowMatrix k_row = k;
    const int q_size = static_cast<int>(q_row.size());
    const int k_size = static_cast<int>(k_row.size());
    const int half_dim = static_cast<int>(q.cols()) / 2;

    std::vector<float> cos_row(half_dim);
    std::vector<float> sin_row(half_dim);
    for (int i = 0; i < half_dim; ++i) {
        cos_row[i] = cos_cache(pos, i);
        sin_row[i] = sin_cache(pos, i);
    }

    ctx->scratch_q.copy_from_host(q_row.data(), q_size);
    ctx->scratch_k.copy_from_host(k_row.data(), k_size);
    ctx->scratch_cos.copy_from_host(cos_row.data(), half_dim);
    ctx->scratch_sin.copy_from_host(sin_row.data(), half_dim);

    const int work = static_cast<int>(q.rows() + k.rows()) * half_dim;
    const int threads = 128;
    const int blocks = (work + threads - 1) / threads;
    rope_row_kernel<<<blocks, threads>>>(ctx->scratch_q.get(), ctx->scratch_k.get(),
                                         static_cast<int>(q.rows()), static_cast<int>(k.rows()),
                                         static_cast<int>(q.cols()),
                                         ctx->scratch_cos.get(), ctx->scratch_sin.get());
    check_cuda(cudaGetLastError(), "rope_cached kernel launch");
    check_cuda(cudaDeviceSynchronize(), "rope_cached kernel sync");

    ctx->scratch_q.copy_to_host(q_row.data(), q_size);
    ctx->scratch_k.copy_to_host(k_row.data(), k_size);
    q = q_row;
    k = k_row;
}

void cuda_write_kv_cache(CudaForwardContext* ctx,
                          int layer, int pos,
                          const float* k_data, const float* v_data) {
    ctx->write_kv(layer, pos, k_data, v_data);
}

void cuda_attention_cached(CudaForwardContext* ctx,
                            int layer, int seq_len,
                            const float* q_data,
                            int n_heads, int n_kv_heads, int head_dim,
                            Eigen::VectorXf& output) {
    if (seq_len <= 0) {
        throw std::invalid_argument("cuda_attention_cached seq_len must be positive");
    }
    if (layer < 0 || layer >= ctx->n_layers) {
        throw std::invalid_argument("cuda_attention_cached layer out of range");
    }

    output.resize(n_heads * head_dim);

    const size_t q_count = static_cast<size_t>(n_heads) * head_dim;

    ctx->scratch_q.copy_from_host(q_data, q_count);

    const int threads = 256;
    const size_t shared_bytes =
        (static_cast<size_t>(seq_len) + head_dim + threads) * sizeof(float);
    attention_kernel<<<n_heads, threads, shared_bytes>>>(
        ctx->scratch_q.get(), ctx->k(layer), ctx->v(layer), ctx->scratch_attn.get(),
        seq_len, ctx->max_seq_len * head_dim, head_dim, n_heads, n_kv_heads);
    check_cuda(cudaGetLastError(), "attention_kernel launch");
    check_cuda(cudaDeviceSynchronize(), "attention_kernel sync");

    ctx->scratch_attn.copy_to_host(output.data(), q_count);
}

}  // namespace backend
}  // namespace llama
