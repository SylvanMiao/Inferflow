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

# ---- Engine library ----
set(ENGINE_SOURCES
    src/ops.cpp
    src/kv_cache.cpp
    src/sampler.cpp
    src/engine.cpp
)

set(ENGINE_HEADERS
    include/llama/config.h
    include/llama/weights.h
    include/llama/ops.h
    include/llama/kv_cache.h
    include/llama/sampler.h
    include/llama/engine.h
)

add_library(llama_engine STATIC ${ENGINE_SOURCES} ${ENGINE_HEADERS})
target_include_directories(llama_engine PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${EIGEN_DIR}
)
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
