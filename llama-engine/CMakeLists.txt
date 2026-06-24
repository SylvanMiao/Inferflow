cmake_minimum_required(VERSION 3.16)
project(llama_engine LANGUAGES CXX)

option(INFERFLOW_ENABLE_CUDA "Build optional CUDA backend tests" OFF)
option(INFERFLOW_ENABLE_AVX2 "Build CPU Int8 SIMD path with AVX2 when supported" ON)

if(INFERFLOW_ENABLE_CUDA)
    if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
        set(CMAKE_CUDA_ARCHITECTURES 75 86 89)
    endif()
    enable_language(CUDA)
    find_package(CUDAToolkit REQUIRED)
endif()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    add_compile_options(-Wno-unknown-pragmas)
endif()

include(CheckCXXCompilerFlag)
find_package(OpenMP)
if(INFERFLOW_ENABLE_AVX2 AND CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64|AMD64|amd64")
    check_cxx_compiler_flag("-mavx2" INFERFLOW_COMPILER_SUPPORTS_AVX2)
endif()

# ---- Eigen (header-only, bundled) ----
set(EIGEN_DIR "${CMAKE_CURRENT_SOURCE_DIR}/third_party/eigen3")
if(NOT EXISTS "${EIGEN_DIR}/Eigen/Core")
    message(FATAL_ERROR "Eigen not found at ${EIGEN_DIR}. Run: wget ... && tar xf eigen.tar.gz -C third_party/")
endif()

# ---- SentencePiece tokenizer ----
find_path(SENTENCEPIECE_INCLUDE_DIR
    NAMES sentencepiece_processor.h
    PATHS /usr/local/include /usr/include
)
find_library(SENTENCEPIECE_LIBRARY
    NAMES sentencepiece
    PATHS /usr/local/lib /usr/lib /usr/lib/x86_64-linux-gnu
)
if(NOT SENTENCEPIECE_INCLUDE_DIR OR NOT SENTENCEPIECE_LIBRARY)
    message(FATAL_ERROR "SentencePiece not found. Expected sentencepiece_processor.h and libsentencepiece.")
endif()

# ---- Engine library ----
set(ENGINE_SOURCES
    src/ops.cpp
    src/kv_cache.cpp
    src/sampler.cpp
    src/tokenizer.cpp
    src/engine.cpp
    src/quantization.cpp
    src/backend/backend.cpp
    src/backend/cpu_backend.cpp
)

set(ENGINE_HEADERS
    include/llama/config.h
    include/llama/weights.h
    include/llama/ops.h
    include/llama/kv_cache.h
    include/llama/sampler.h
    include/llama/tokenizer.h
    include/llama/engine.h
    include/llama/quantization.h
    include/llama/backend/backend.h
    include/llama/backend/cpu_backend.h
)

if(INFERFLOW_ENABLE_CUDA)
    list(APPEND ENGINE_SOURCES src/backend/cuda_backend.cu)
    list(APPEND ENGINE_HEADERS include/llama/backend/cuda_backend.h)
endif()

add_library(llama_engine STATIC ${ENGINE_SOURCES} ${ENGINE_HEADERS})
target_include_directories(llama_engine PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${EIGEN_DIR}
    ${SENTENCEPIECE_INCLUDE_DIR}
)
target_link_libraries(llama_engine PUBLIC ${SENTENCEPIECE_LIBRARY})
target_compile_options(llama_engine PRIVATE -O2 -Wall -Wextra)
if(OpenMP_CXX_FOUND)
    target_link_libraries(llama_engine PUBLIC OpenMP::OpenMP_CXX)
endif()
if(INFERFLOW_ENABLE_AVX2 AND INFERFLOW_COMPILER_SUPPORTS_AVX2)
    target_compile_definitions(llama_engine PRIVATE INFERFLOW_ENABLE_AVX2)
endif()
if(INFERFLOW_ENABLE_CUDA)
    target_link_libraries(llama_engine PUBLIC CUDA::cudart)
    target_compile_definitions(llama_engine PUBLIC INFERFLOW_ENABLE_CUDA)
    target_compile_definitions(llama_engine PRIVATE EIGEN_NO_CUDA)
    set_target_properties(llama_engine PROPERTIES
        CUDA_STANDARD 17
        CUDA_STANDARD_REQUIRED ON
        CUDA_SEPARABLE_COMPILATION OFF
    )
endif()

# ---- Tests ----
add_executable(test_ops test/test_ops.cpp)
target_link_libraries(test_ops llama_engine)
target_include_directories(test_ops PRIVATE ${EIGEN_DIR})

add_executable(test_engine test/test_engine.cpp)
target_link_libraries(test_engine llama_engine)
target_include_directories(test_engine PRIVATE ${EIGEN_DIR})

add_executable(test_forward test/test_forward.cpp)
target_link_libraries(test_forward llama_engine)
target_include_directories(test_forward PRIVATE ${EIGEN_DIR})

add_executable(test_quantization test/test_quantization.cpp)
target_link_libraries(test_quantization llama_engine)
target_include_directories(test_quantization PRIVATE ${EIGEN_DIR})

add_executable(test_int8_forward test/test_int8_forward.cpp)
target_link_libraries(test_int8_forward llama_engine)
target_include_directories(test_int8_forward PRIVATE ${EIGEN_DIR})

add_executable(test_backend test/test_backend.cpp)
target_link_libraries(test_backend llama_engine)
target_include_directories(test_backend PRIVATE ${EIGEN_DIR})

if(INFERFLOW_ENABLE_CUDA)
    add_executable(test_cuda_backend test/test_cuda_backend.cpp)
    target_link_libraries(test_cuda_backend llama_engine)
    target_include_directories(test_cuda_backend PRIVATE ${EIGEN_DIR})

    add_executable(test_cuda_forward test/test_cuda_forward.cpp)
    target_link_libraries(test_cuda_forward llama_engine)
    target_include_directories(test_cuda_forward PRIVATE ${EIGEN_DIR})
endif()

add_executable(bench_forward bench/bench_forward.cpp)
target_link_libraries(bench_forward llama_engine)
target_include_directories(bench_forward PRIVATE ${EIGEN_DIR})

# ---- CLI demo ----
add_executable(inferflow_cli demo/chat_cli.cpp)
target_link_libraries(inferflow_cli llama_engine)
target_compile_definitions(inferflow_cli PRIVATE
    INFERFLOW_DEFAULT_MODEL_PATH="${CMAKE_CURRENT_SOURCE_DIR}/test/tinyllama.bin"
    INFERFLOW_DEFAULT_TOKENIZER_PATH="${CMAKE_CURRENT_SOURCE_DIR}/../../models/tokenizer.model"
)
