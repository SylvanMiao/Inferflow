#pragma once
/**
 * Token sampler — maps 1:1 to Python GenerationConfig + sampling logic.
 *
 * Currently supports:
 *   - Greedy (temperature = 0)
 *   - Temperature scaling
 *   - Top-K filtering
 *   - Top-P (nucleus) filtering
 */

#include <Eigen/Dense>
#include <random>
#include "config.h"

namespace llama {

class Sampler {
public:
    Sampler(const GenerationConfig& config);

    /**
     * Sample the next token from logits.
     *
     * @param logits  [vocab_size] unnormalized log-probabilities
     * @return       sampled token id
     */
    int sample(const Eigen::VectorXf& logits);

    /** Reconfigure sampler parameters (e.g., between requests). */
    void set_config(const GenerationConfig& config) { config_ = config; }

private:
    void apply_temperature(Eigen::VectorXf& logits);
    void apply_top_k(Eigen::VectorXf& logits);
    void apply_top_p(Eigen::VectorXf& logits);

    GenerationConfig config_;
    std::mt19937 rng_;
};

}  // namespace llama
