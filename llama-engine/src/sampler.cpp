/**
 * Token sampler implementation.
 */

#include "llama/sampler.h"
#include <algorithm>
#include <numeric>

namespace llama {

Sampler::Sampler(const GenerationConfig& config) : config_(config) {
    if (config_.seed >= 0) {
        rng_.seed(static_cast<uint32_t>(config_.seed));
    } else {
        std::random_device rd;
        rng_.seed(rd());
    }
}

int Sampler::sample(const Eigen::VectorXf& logits) {
    Eigen::VectorXf probs = logits;

    // Temperature = 0 → greedy
    if (config_.temperature <= 0.0f) {
        Eigen::Index max_idx;
        probs.maxCoeff(&max_idx);
        return static_cast<int>(max_idx);
    }

    apply_temperature(probs);
    apply_top_k(probs);
    apply_top_p(probs);

    // Softmax
    float max_val = probs.maxCoeff();
    probs = (probs.array() - max_val).exp();
    float sum = probs.sum();
    if (sum > 0.0f) {
        probs /= sum;

        // Sample from categorical distribution
        std::uniform_real_distribution<float> dist(0.0f, 1.0f);
        float r = dist(rng_);
        float cumulative = 0.0f;
        for (int i = 0; i < probs.size(); ++i) {
            cumulative += probs(i);
            if (r <= cumulative) {
                return i;
            }
        }
    }

    // Fallback: argmax
    Eigen::Index max_idx;
    probs.maxCoeff(&max_idx);
    return static_cast<int>(max_idx);
}

void Sampler::apply_temperature(Eigen::VectorXf& logits) {
    if (config_.temperature > 0.0f) {
        logits /= config_.temperature;
    }
}

void Sampler::apply_top_k(Eigen::VectorXf& logits) {
    int k = config_.top_k;
    int vocab_size = static_cast<int>(logits.size());
    if (k <= 0 || k >= vocab_size) return;

    // Find the k-th largest value
    std::vector<float> vals(logits.data(), logits.data() + vocab_size);
    std::nth_element(vals.begin(), vals.begin() + k - 1, vals.end(),
                     std::greater<float>());
    float threshold = vals[k - 1];

    // Mask values below threshold
    for (int i = 0; i < vocab_size; ++i) {
        if (logits(i) < threshold) {
            logits(i) = -std::numeric_limits<float>::infinity();
        }
    }
}

void Sampler::apply_top_p(Eigen::VectorXf& logits) {
    float p = config_.top_p;
    if (p >= 1.0f) return;

    int vocab_size = static_cast<int>(logits.size());

    // Sort indices by logit value (descending)
    std::vector<int> indices(vocab_size);
    std::iota(indices.begin(), indices.end(), 0);
    std::sort(indices.begin(), indices.end(), [&](int a, int b) {
        return logits(a) > logits(b);
    });

    // Compute softmax probabilities
    float max_val = logits.maxCoeff();
    std::vector<float> probs(vocab_size);
    float total = 0.0f;
    for (int i = 0; i < vocab_size; ++i) {
        probs[i] = std::exp(logits(i) - max_val);
        total += probs[i];
    }
    for (int i = 0; i < vocab_size; ++i) {
        probs[i] /= total;
    }

    // Cumulative sum in sorted order
    float cumsum = 0.0f;
    int cutoff = vocab_size;
    for (int i = 0; i < vocab_size; ++i) {
        cumsum += probs[indices[i]];
        if (cumsum > p) {
            cutoff = i + 1;
            break;
        }
    }

    // Mask tokens beyond cutoff
    if (cutoff < vocab_size) {
        for (int i = cutoff; i < vocab_size; ++i) {
            logits(indices[i]) = -std::numeric_limits<float>::infinity();
        }
    }
}

}  // namespace llama
