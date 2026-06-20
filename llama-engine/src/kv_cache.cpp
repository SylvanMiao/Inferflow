/**
 * KV Cache implementation.
 */

#include "llama/kv_cache.h"

namespace llama {

KVCache::KVCache(const ModelConfig& config)
    : n_layers_(config.n_layers)
    , max_seq_len_(config.max_seq_len)
    , kv_dim_(config.kv_dim)
    , current_len_(0) {
    k_cache_.resize(n_layers_);
    v_cache_.resize(n_layers_);
    for (int i = 0; i < n_layers_; ++i) {
        k_cache_[i] = Eigen::MatrixXf::Zero(max_seq_len_, kv_dim_);
        v_cache_[i] = Eigen::MatrixXf::Zero(max_seq_len_, kv_dim_);
    }
}

void KVCache::write(int layer_idx, int pos,
                    const Eigen::VectorXf& k,
                    const Eigen::VectorXf& v) {
    k_cache_[layer_idx].row(pos) = k;
    v_cache_[layer_idx].row(pos) = v;
}

void KVCache::clear() {
    for (int i = 0; i < n_layers_; ++i) {
        k_cache_[i].setZero();
        v_cache_[i].setZero();
    }
    current_len_ = 0;
}

}  // namespace llama
