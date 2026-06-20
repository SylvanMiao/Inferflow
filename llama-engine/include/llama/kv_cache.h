#pragma once
/**
 * Key-Value Cache — maps 1:1 to Python KVCache.
 *
 * Pre-allocates contiguous memory for all layers at init time.
 * During inference, each layer writes its K and V at position `pos`,
 * and reads back all cached K,V for attention computation.
 */

#include <Eigen/Dense>
#include <vector>
#include "config.h"

namespace llama {

class KVCache {
public:
    KVCache(const ModelConfig& config);

    /**
     * Write a single token's K and V at the given position.
     * @param layer_idx  which transformer layer
     * @param pos        token position in the sequence
     * @param k          key vector   [kv_dim]
     * @param v          value vector [kv_dim]
     */
    void write(int layer_idx, int pos,
               const Eigen::VectorXf& k,
               const Eigen::VectorXf& v);

    /**
     * Get all cached K and V for positions [0, seq_len).
     * Returns matrices shaped [seq_len, kv_dim].
     */
    const Eigen::MatrixXf& k(int layer_idx) const { return k_cache_[layer_idx]; }
    const Eigen::MatrixXf& v(int layer_idx) const { return v_cache_[layer_idx]; }

    /**
     * Number of tokens currently cached (managed by caller via pos).
     */
    int current_len() const { return current_len_; }
    void set_current_len(int len) { current_len_ = len; }

    /** Reset all cached values to zero (for a new conversation). */
    void clear();

    int n_layers()    const { return n_layers_; }
    int max_seq_len() const { return max_seq_len_; }
    int kv_dim()      const { return kv_dim_; }

private:
    int n_layers_;
    int max_seq_len_;
    int kv_dim_;
    int current_len_ = 0;

    // k_cache_[layer] shape: [max_seq_len, kv_dim]
    // Each row stores the key vector for one position.
    std::vector<Eigen::MatrixXf> k_cache_;
    std::vector<Eigen::MatrixXf> v_cache_;
};

}  // namespace llama
