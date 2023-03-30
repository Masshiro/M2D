cmake_minimum_required(VERSION 3.16)

project(CardinalityMomentEstimator)

set(CMAKE_CXX_STANDARD 11)

#   Let CMake generate a JSON file containing compile info like include paths.
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

add_definitions("-std=c++11")
add_definitions('-g')
add_definitions('-Wall')

#   conan standard settings
#   variable ${CONAN_LIBS} indicates all of the depending libraries.
include(${CMAKE_BINARY_DIR}/conanbuildinfo.cmake)
conan_basic_setup()

#   Enable unit testings.
enable_testing()

add_subdirectory(src)
add_subdirectory(test)
add_subdirectory(experiment)

add_executable(main main.cpp)
