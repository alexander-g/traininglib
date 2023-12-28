cmake_minimum_required(VERSION 3.0 FATAL_ERROR)
project(torchscript_adapter)

#suppress a warning regarding FetchContent
cmake_policy(SET CMP0135 NEW)

set(MINIZ_DIR ${CMAKE_SOURCE_DIR}/thirdparty/miniz)
add_subdirectory(${MINIZ_DIR})

#add_compile_definitions(MINIZ_NO_STDIO)

target_compile_options(miniz PRIVATE -fPIC)



include(FetchContent)
set(JSON_URL https://github.com/nlohmann/json/releases/download/v3.11.3/json.tar.xz)
message("Maybe downloading ${JSON_URL}")
FetchContent_Declare(json 
    URL ${JSON_URL}
)
FetchContent_MakeAvailable(json)


if (UNIX)
    #NOTE: pre-cxx11 abi required
    set(PYTORCH_URL https://download.pytorch.org/libtorch/cpu/libtorch-shared-with-deps-2.1.2%2Bcpu.zip)
elseif (WIN32)
    set(PYTORCH_URL https://download.pytorch.org/libtorch/cpu/libtorch-win-shared-with-deps-2.1.2%2Bcpu.zip)
endif()

message("Maybe downloading ${PYTORCH_URL}")
FetchContent_Declare(pytorch
    URL ${PYTORCH_URL}
)
FetchContent_MakeAvailable(pytorch)
find_package(Torch REQUIRED HINTS "${CMAKE_BINARY_DIR}/_deps/pytorch-src/share/cmake/Torch")


add_library(TSinterface SHARED interface.cpp miniz.cpp)
set_property(TARGET TSinterface PROPERTY CXX_STANDARD 17)
target_include_directories(TSinterface PRIVATE ${MINIZ_DIR})
target_include_directories(TSinterface PRIVATE ${CMAKE_BINARY_DIR}/thirdparty/miniz)
target_link_libraries(TSinterface PRIVATE miniz)
add_dependencies(TSinterface miniz)
target_include_directories(TSinterface PRIVATE ${CMAKE_BINARY_DIR}/_deps/json-src/single_include/)
target_include_directories(TSinterface PRIVATE ${CMAKE_BINARY_DIR}/_deps/pytorch-src/include/)
target_link_libraries(TSinterface PRIVATE "${TORCH_LIBRARIES}")
