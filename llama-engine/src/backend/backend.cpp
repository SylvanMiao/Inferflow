#include "llama/backend/backend.h"

namespace llama {
namespace backend {

const char* backend_name(BackendKind backend) {
    switch (backend) {
        case BackendKind::CPU: return "cpu";
        case BackendKind::CUDA: return "cuda";
    }
    return "unknown";
}

}  // namespace backend
}  // namespace llama
