cmake_minimum_required(VERSION 3.13)
project(test)
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)


add_executable(test
  main.cpp 
)
set(DEP_DIR "/home/boris/il_dependencies/test/")
find_package(PkgConfig REQUIRED)
pkg_search_module(PROTOBUF REQUIRED protobuf)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -pthread")

target_include_directories(test BEFORE PRIVATE ${PROTOBUF_INCLUDE_DIRS})
target_include_directories(test BEFORE PRIVATE "${DEP_DIR}include")
target_link_libraries(test PRIVATE "${PROTOBUF_LIBRARY_DIRS}/libprotobuf.a")
target_link_libraries(test PRIVATE "${DEP_DIR}lib/libonnx_proto.a") 
target_link_libraries(test PRIVATE "${DEP_DIR}lib/libonnx.a") 


