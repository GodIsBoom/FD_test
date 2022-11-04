PROJECT(infer_demo C CXX)
CMAKE_MINIMUM_REQUIRED (VERSION 3.12)

include(fastdeploy-linux-x64-0.4.0/FastDeploy.cmake)

# Add FastDeploy dependency header files
include_directories(${FASTDEPLOY_INCS})

add_executable(infer_demo ${PROJECT_SOURCE_DIR}/infer_demo.cc)
target_link_libraries(infer_demo ${FASTDEPLOY_LIBS})