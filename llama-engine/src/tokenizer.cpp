#include "llama/tokenizer.h"

#include <sentencepiece_processor.h>
#include <iostream>

namespace llama {

class SentencePieceTokenizer::Impl {
public:
    sentencepiece::SentencePieceProcessor processor;
    bool loaded = false;
};

SentencePieceTokenizer::SentencePieceTokenizer() : impl_(std::make_unique<Impl>()) {}

SentencePieceTokenizer::SentencePieceTokenizer(const std::string& path)
    : SentencePieceTokenizer() {
    load(path);
}

SentencePieceTokenizer::~SentencePieceTokenizer() = default;

bool SentencePieceTokenizer::load(const std::string& path) {
    auto status = impl_->processor.Load(path);
    if (!status.ok()) {
        std::cerr << "Failed to load SentencePiece model: " << path
                  << " (" << status.ToString() << ")" << std::endl;
        impl_->loaded = false;
        return false;
    }

    impl_->loaded = true;
    return true;
}

std::vector<int> SentencePieceTokenizer::encode(const std::string& text,
                                                bool add_bos,
                                                bool add_eos) const {
    std::vector<int> token_ids;
    if (!impl_->loaded) return token_ids;

    token_ids = impl_->processor.EncodeAsIds(text);
    if (add_bos) {
        token_ids.insert(token_ids.begin(), bos_id());
    }
    if (add_eos) {
        token_ids.push_back(eos_id());
    }

    return token_ids;
}

std::string SentencePieceTokenizer::decode(int token_id) const {
    if (!impl_->loaded) return {};
    std::vector<int> token_ids{token_id};
    return impl_->processor.DecodeIds(token_ids);
}

std::string SentencePieceTokenizer::decode(const std::vector<int>& token_ids) const {
    if (!impl_->loaded) return {};
    return impl_->processor.DecodeIds(token_ids);
}

bool SentencePieceTokenizer::is_eos(int token_id) const {
    return impl_->loaded && token_id == eos_id();
}

int SentencePieceTokenizer::vocab_size() const {
    if (!impl_->loaded) return 0;
    return impl_->processor.GetPieceSize();
}

int SentencePieceTokenizer::bos_id() const {
    if (!impl_->loaded) return -1;
    return impl_->processor.bos_id();
}

int SentencePieceTokenizer::eos_id() const {
    if (!impl_->loaded) return -1;
    return impl_->processor.eos_id();
}

}  // namespace llama
