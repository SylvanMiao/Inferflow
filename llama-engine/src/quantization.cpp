#include "llama/quantization.h"

#include <algorithm>
#include <cmath>
#include <stdexcept>

#ifdef _OPENMP
#include <omp.h>
#endif

#ifdef INFERFLOW_ENABLE_AVX2
#include <immintrin.h>
#if defined(__GNUC__) || defined(__clang__)
#define INFERFLOW_AVX2_TARGET __attribute__((target("avx2")))
#else
#define INFERFLOW_AVX2_TARGET
#endif
#endif

namespace llama {
namespace quant {

namespace {
int ceil_div(int a, int b) {
    return (a + b - 1) / b;
}

#ifdef INFERFLOW_ENABLE_AVX2
INFERFLOW_AVX2_TARGET float horizontal_sum(__m256 value) {
    alignas(32) float lanes[8];
    _mm256_store_ps(lanes, value);
    return lanes[0] + lanes[1] + lanes[2] + lanes[3]
         + lanes[4] + lanes[5] + lanes[6] + lanes[7];
}

INFERFLOW_AVX2_TARGET float dot_dequant_avx2(const QuantizedMatrixInt8& qweight,
                                             const Eigen::VectorXf& input,
                                             int row) {
    const int row_offset = row * qweight.cols;
    int c = 0;
    __m256 acc = _mm256_setzero_ps();

    for (; c + 8 <= qweight.cols; c += 8) {
        const int idx = row_offset + c;
        const int group = idx / qweight.group_size;
        const int last_group = (idx + 7) / qweight.group_size;
        if (group != last_group) {
            break;
        }

        const __m128i packed_i8 =
            _mm_loadl_epi64(reinterpret_cast<const __m128i*>(qweight.values.data() + idx));
        const __m256i weight_i32 = _mm256_cvtepi8_epi32(packed_i8);
        const __m256 weight_f32 = _mm256_cvtepi32_ps(weight_i32);
        const __m256 input_f32 = _mm256_loadu_ps(input.data() + c);
        const __m256 scale = _mm256_set1_ps(qweight.scales[group]);
        acc = _mm256_add_ps(acc, _mm256_mul_ps(_mm256_mul_ps(weight_f32, scale), input_f32));
    }

    float sum = horizontal_sum(acc);
    for (; c < qweight.cols; ++c) {
        const int idx = row_offset + c;
        const int group = idx / qweight.group_size;
        sum += input(c) * static_cast<float>(qweight.values[idx]) * qweight.scales[group];
    }
    return sum;
}
#endif
}  // namespace

int QuantizedMatrixInt8::num_groups() const {
    if (group_size <= 0) return 0;
    return ceil_div(rows * cols, group_size);
}

std::string QuantizedMatrixInt8::validate() const {
    if (rows <= 0) return "rows must be positive";
    if (cols <= 0) return "cols must be positive";
    if (group_size <= 0) return "group_size must be positive";
    if (static_cast<int>(values.size()) != rows * cols) return "values size mismatch";
    if (static_cast<int>(scales.size()) != num_groups()) return "scales size mismatch";
    return {};
}

QuantizedMatrixInt8 quantize_groupwise(const Eigen::MatrixXf& weight, int group_size) {
    if (weight.rows() <= 0 || weight.cols() <= 0) {
        throw std::invalid_argument("weight must be non-empty");
    }
    if (group_size <= 0) {
        throw std::invalid_argument("group_size must be positive");
    }

    QuantizedMatrixInt8 q;
    q.rows = static_cast<int>(weight.rows());
    q.cols = static_cast<int>(weight.cols());
    q.group_size = group_size;
    q.values.resize(q.rows * q.cols);
    q.scales.resize(q.num_groups(), 1.0f);

    for (int g = 0; g < q.num_groups(); ++g) {
        const int begin = g * group_size;
        const int end = std::min(begin + group_size, q.rows * q.cols);

        float abs_max = 0.0f;
        for (int idx = begin; idx < end; ++idx) {
            const int r = idx / q.cols;
            const int c = idx % q.cols;
            abs_max = std::max(abs_max, std::abs(weight(r, c)));
        }

        const float scale = abs_max > 0.0f ? abs_max / 127.0f : 1.0f;
        q.scales[g] = scale;

        for (int idx = begin; idx < end; ++idx) {
            const int r = idx / q.cols;
            const int c = idx % q.cols;
            const float scaled = weight(r, c) / scale;
            const int rounded = static_cast<int>(std::nearbyint(scaled));
            const int clamped = std::max(-127, std::min(127, rounded));
            q.values[idx] = static_cast<int8_t>(clamped);
        }
    }

    return q;
}

Eigen::MatrixXf dequantize_groupwise(const QuantizedMatrixInt8& qweight) {
    const std::string error = qweight.validate();
    if (!error.empty()) {
        throw std::invalid_argument("invalid QuantizedMatrixInt8: " + error);
    }

    Eigen::MatrixXf weight(qweight.rows, qweight.cols);
    for (int idx = 0; idx < qweight.rows * qweight.cols; ++idx) {
        const int r = idx / qweight.cols;
        const int c = idx % qweight.cols;
        const int group = idx / qweight.group_size;
        weight(r, c) = static_cast<float>(qweight.values[idx]) * qweight.scales[group];
    }
    return weight;
}

void matmul_dequant(const QuantizedMatrixInt8& qweight,
                    const Eigen::VectorXf& input,
                    Eigen::VectorXf& output) {
    const std::string error = qweight.validate();
    if (!error.empty()) {
        throw std::invalid_argument("invalid QuantizedMatrixInt8: " + error);
    }
    if (input.size() != qweight.cols) {
        throw std::invalid_argument("input size does not match qweight cols");
    }

    output.resize(qweight.rows);
    #pragma omp parallel for if(qweight.rows >= 4) schedule(static)
    for (int r = 0; r < qweight.rows; ++r) {
#ifdef INFERFLOW_ENABLE_AVX2
        output(r) = dot_dequant_avx2(qweight, input, r);
#else
        float sum = 0.0f;
        const int row_offset = r * qweight.cols;
        for (int c = 0; c < qweight.cols; ++c) {
            const int idx = row_offset + c;
            const int group = idx / qweight.group_size;
            sum += input(c) * static_cast<float>(qweight.values[idx]) * qweight.scales[group];
        }
        output(r) = sum;
#endif
    }
}

Eigen::VectorXf matmul_dequant(const QuantizedMatrixInt8& qweight,
                               const Eigen::VectorXf& input) {
    Eigen::VectorXf output;
    matmul_dequant(qweight, input, output);
    return output;
}

}  // namespace quant
}  // namespace llama
