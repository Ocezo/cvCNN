cmake_minimum_required(VERSION 3.18)

# CUDA configuration matching the installed LibTorch 2.5.1+cu121.
# These cache defaults can still be overridden with -D options if needed.
set(CUDA_TOOLKIT_ROOT_DIR "/usr/local/cuda-12.1" CACHE PATH
    "Path to the CUDA toolkit")
set(CMAKE_CUDA_COMPILER "${CUDA_TOOLKIT_ROOT_DIR}/bin/nvcc" CACHE FILEPATH
    "Path to the CUDA compiler")
set(CMAKE_CUDA_ARCHITECTURES "86" CACHE STRING
    "CUDA architectures to target (86 = GeForce RTX 3060)")

project(cvCNN LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(OpenCV REQUIRED)
set(Torch_DIR "/opt/libtorch/share/cmake/Torch")
find_package(Torch REQUIRED)

add_executable(cvCNN
    src/main.cpp
)

target_include_directories(cvCNN PRIVATE
    ${PROJECT_SOURCE_DIR}/include
)

target_link_libraries(cvCNN PRIVATE
    ${OpenCV_LIBS}
    ${TORCH_LIBRARIES}
)

target_compile_options(cvCNN PRIVATE
    ${TORCH_CXX_FLAGS}
)

set_target_properties(cvCNN PROPERTIES
    BUILD_RPATH "/opt/libtorch/lib"
)
