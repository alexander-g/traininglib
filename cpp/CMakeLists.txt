cmake_minimum_required(VERSION 3.0 FATAL_ERROR)
project(torchscript_adapter)

#suppress a warning regarding FetchContent
cmake_policy(SET CMP0135 NEW)

set(MINIZ_DIR ${CMAKE_SOURCE_DIR}/thirdparty/miniz)
add_subdirectory(${MINIZ_DIR})

#add_compile_definitions(MINIZ_NO_STDIO)

target_compile_options(miniz PRIVATE -fPIC)



include(FetchContent)
FetchContent_Declare(json 
    URL https://github.com/nlohmann/json/releases/download/v3.11.3/json.tar.xz
)
FetchContent_MakeAvailable(json)



add_executable(test interface.cpp)
set_property(TARGET test PROPERTY CXX_STANDARD 17)
target_include_directories(test PRIVATE ${MINIZ_DIR})
target_include_directories(test PRIVATE ${CMAKE_BINARY_DIR}/thirdparty/miniz)
target_link_libraries(test PRIVATE miniz)
add_dependencies(test miniz)
target_include_directories(test PRIVATE ${CMAKE_BINARY_DIR}/_deps/json-src/single_include/)

#target_link_libraries(test PRIVATE nlohmann_json::nlohmann_json)

#target_link_libraries(test "${TORCH_LIBRARIES}")



add_library(testlib SHARED interface.cpp)
target_include_directories(testlib PRIVATE ${MINIZ_DIR})
target_include_directories(testlib PRIVATE ${CMAKE_BINARY_DIR}/thirdparty/miniz)
target_link_libraries(testlib PRIVATE miniz)
add_dependencies(testlib miniz)
target_include_directories(testlib PRIVATE ${CMAKE_BINARY_DIR}/_deps/json-src/single_include/)
