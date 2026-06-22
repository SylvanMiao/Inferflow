cmake_minimum_required(VERSION 3.16)
project(llama_engine LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

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
)

set(ENGINE_HEADERS
    include/llama/config.h
    include/llama/weights.h
    include/llama/ops.h
    include/llama/kv_cache.h
    include/llama/sampler.h
    include/llama/tokenizer.h
    include/llama/engine.h
)

add_library(llama_engine STATIC ${ENGINE_SOURCES} ${ENGINE_HEADERS})
target_include_directories(llama_engine PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${EIGEN_DIR}
    ${SENTENCEPIECE_INCLUDE_DIR}
)
target_link_libraries(llama_engine PUBLIC ${SENTENCEPIECE_LIBRARY})
target_compile_options(llama_engine PRIVATE -O2 -Wall -Wextra)

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

# ---- CLI demo ----
add_executable(inferflow_cli demo/chat_cli.cpp)
target_link_libraries(inferflow_cli llama_engine)
target_compile_definitions(inferflow_cli PRIVATE
    INFERFLOW_DEFAULT_MODEL_PATH="${CMAKE_CURRENT_SOURCE_DIR}/test/tinyllama.bin"
    INFERFLOW_DEFAULT_TOKENIZER_PATH="${CMAKE_CURRENT_SOURCE_DIR}/../../models/tokenizer.model"
)
