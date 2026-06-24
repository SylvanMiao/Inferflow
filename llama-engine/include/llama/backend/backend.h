#pragma once

namespace llama {
namespace backend {

enum class BackendKind {
    CPU,
    CUDA,
};

const char* backend_name(BackendKind backend);

}  // namespace backend
}  // namespace llama
