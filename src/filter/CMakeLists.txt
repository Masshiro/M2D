#   ./src/filter/CMakeLists.txt
add_library(MinheapFilter MinheapFilter.hpp)
target_link_libraries(MinheapFilter ${CONAN_LIBS} leader_zero)

add_library(MapImplFilter MapImplFilter.hpp)
target_link_libraries(MapImplFilter ${CONAN_LIBS} leader_zero)