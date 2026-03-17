cmake_minimum_required(VERSION 3.18)
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
