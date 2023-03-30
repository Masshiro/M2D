#   ./src/distribution/CMakeLists.txt
add_library(HeadTail HeadTail.hpp)
target_link_libraries(HeadTail ${CONAN_LIBS} leader_zero)