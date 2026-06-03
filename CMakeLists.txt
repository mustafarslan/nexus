cmake_minimum_required(VERSION 3.25)
project(nexus VERSION 0.1.0 LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Automatically apply the submodule patch before building
find_package(Git REQUIRED)
message(STATUS "Checking llama.cpp patch status...")
execute_process(
    COMMAND ${GIT_EXECUTABLE} apply --reverse --check "${CMAKE_CURRENT_SOURCE_DIR}/scripts/llama_kv_ring_buffer.patch"
    WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/external/llama.cpp"
    RESULT_VARIABLE PATCH_REVERSE_CHECK_RES
    OUTPUT_QUIET
    ERROR_QUIET
)

if (PATCH_REVERSE_CHECK_RES EQUAL 0)
    message(STATUS "llama_kv_ring_buffer.patch is already applied.")
else()
    execute_process(
        COMMAND ${GIT_EXECUTABLE} apply --check "${CMAKE_CURRENT_SOURCE_DIR}/scripts/llama_kv_ring_buffer.patch"
        WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/external/llama.cpp"
        RESULT_VARIABLE PATCH_CHECK_RES
        OUTPUT_QUIET
        ERROR_QUIET
    )
    if (PATCH_CHECK_RES EQUAL 0)
        message(STATUS "Applying llama_kv_ring_buffer.patch...")
        execute_process(
            COMMAND ${GIT_EXECUTABLE} apply "${CMAKE_CURRENT_SOURCE_DIR}/scripts/llama_kv_ring_buffer.patch"
            WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/external/llama.cpp"
            RESULT_VARIABLE PATCH_APPLY_RES
        )
        if (NOT PATCH_APPLY_RES EQUAL 0)
            message(WARNING "Failed to apply llama_kv_ring_buffer.patch")
        endif()
    else()
        message(WARNING "llama_kv_ring_buffer.patch cannot be applied cleanly (and is not already applied).")
    endif()
endif()

# Add llama.cpp as a subdirectory (already checked out at pinned tag b3400)
add_subdirectory(external/llama.cpp)

# Target for the Offline KV Compiler CLI
add_executable(nexus_kv_compiler
    src/nexus_kv_compiler.cpp
)
target_compile_features(nexus_kv_compiler PRIVATE cxx_std_23)

target_include_directories(nexus_kv_compiler PRIVATE
    src/
    external/llama.cpp/include/
    external/llama.cpp/src/          # For internal headers (llama-kv-cache.h, llama-context.h, llama-impl.h)
    external/llama.cpp/ggml/include/ # For ggml.h, ggml-backend.h
)

target_link_libraries(nexus_kv_compiler PRIVATE
    llama
    ggml
)

# Target for verification utility
add_executable(verify_atb
    test/verify_atb.cpp
)
target_compile_features(verify_atb PRIVATE cxx_std_23)

target_include_directories(verify_atb PRIVATE
    src/
)

# Find Python and Nanobind to build Python extension bindings
find_package(Python 3.8 REQUIRED COMPONENTS Interpreter Development.Module)

execute_process(
    COMMAND "${Python_EXECUTABLE}" -m nanobind --cmake_dir
    OUTPUT_VARIABLE nanobind_cmake_dir
    OUTPUT_STRIP_TRAILING_WHITESPACE
)
list(APPEND CMAKE_PREFIX_PATH "${nanobind_cmake_dir}")
find_package(nanobind REQUIRED)

# Nanobind binary module for FSM logit processing and KV cache splicing
nanobind_add_module(nexus_fsm_ext
    src/nexus_fsm.cpp
    src/bindings.cpp
    src/nexus_page_mounter.cpp
    src/nexus_os_compat.cpp
    src/nexus_kv_splicer.cpp
    src/nexus_slb.cpp
    src/nexus_orchestrator.cpp
    src/nexus_rope_math.cpp
    src/nexus_block_cache.cpp
)

if (CMAKE_CXX_COMPILER_ID MATCHES "Clang" OR CMAKE_CXX_COMPILER_ID MATCHES "GNU" OR CMAKE_CXX_COMPILER_ID MATCHES "AppleClang")
    target_compile_options(nexus_fsm_ext PRIVATE -O3)
endif()

target_compile_features(nexus_fsm_ext PRIVATE cxx_std_23)
target_include_directories(nexus_fsm_ext PRIVATE
    src/
    external/llama.cpp/include/
    external/llama.cpp/src/          # For internal headers
    external/llama.cpp/ggml/include/ # For ggml.h
)

target_link_libraries(nexus_fsm_ext PRIVATE
    llama
    ggml
)

# Platform-specific framework linkage for macOS (Metal)
if(APPLE)
    target_link_libraries(nexus_kv_compiler PRIVATE "-framework Metal" "-framework Foundation")
    target_link_libraries(verify_atb PRIVATE "-framework Metal" "-framework Foundation")
    target_link_libraries(nexus_fsm_ext PRIVATE "-framework Metal" "-framework Foundation")
endif()

# Target for the TTFT C++ microbenchmark
find_package(benchmark QUIET)
if (benchmark_FOUND)
    add_executable(bench_phase21_ttft
        test/bench_phase21_ttft.cpp
        src/nexus_os_compat.cpp
        src/nexus_block_cache.cpp
        src/nexus_rope_math.cpp
        src/nexus_kv_splicer.cpp
    )
    target_compile_features(bench_phase21_ttft PRIVATE cxx_std_23)
    target_include_directories(bench_phase21_ttft PRIVATE
        src/
        external/llama.cpp/include/
        external/llama.cpp/src/
        external/llama.cpp/ggml/include/
    )
    target_link_libraries(bench_phase21_ttft PRIVATE
        llama
        ggml
        benchmark::benchmark
    )
    if(APPLE)
        target_link_libraries(bench_phase21_ttft PRIVATE "-framework Metal" "-framework Foundation")
    endif()
else()
    # Standalone benchmark executable without google/benchmark dependency
    add_executable(bench_phase21_ttft
        test/bench_phase21_ttft.cpp
        src/nexus_os_compat.cpp
        src/nexus_block_cache.cpp
        src/nexus_rope_math.cpp
        src/nexus_kv_splicer.cpp
    )
    target_compile_features(bench_phase21_ttft PRIVATE cxx_std_23)
    target_include_directories(bench_phase21_ttft PRIVATE
        src/
        external/llama.cpp/include/
        external/llama.cpp/src/
        external/llama.cpp/ggml/include/
    )
    target_link_libraries(bench_phase21_ttft PRIVATE
        llama
        ggml
    )
    if(APPLE)
        target_link_libraries(bench_phase21_ttft PRIVATE "-framework Metal" "-framework Foundation")
    endif()
endif()
