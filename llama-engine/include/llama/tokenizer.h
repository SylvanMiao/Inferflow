#pragma once

#include <memory>
#include <string>
#include <vector>

namespace llama {

class Tokenizer {
public:
    virtual ~Tokenizer() = default;

    virtual bool load(const std::string& path) = 0;
    virtual std::vector<int> encode(const std::string& text,
                                    bool add_bos = true,
                                    bool add_eos = false) const = 0;
    virtual std::string decode(int token_id) const = 0;
    virtual std::string decode(const std::vector<int>& token_ids) const = 0;
    virtual bool is_eos(int token_id) const = 0;
    virtual int vocab_size() const = 0;
    virtual int bos_id() const = 0;
    virtual int eos_id() const = 0;
};

class SentencePieceTokenizer final : public Tokenizer {
public:
    SentencePieceTokenizer();
    explicit SentencePieceTokenizer(const std::string& path);
    ~SentencePieceTokenizer() override;

    bool load(const std::string& path) override;
    std::vector<int> encode(const std::string& text,
                            bool add_bos = true,
                            bool add_eos = false) const override;
    std::string decode(int token_id) const override;
    std::string decode(const std::vector<int>& token_ids) const override;
    bool is_eos(int token_id) const override;
    int vocab_size() const override;
    int bos_id() const override;
    int eos_id() const override;

private:
    class Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace llama
