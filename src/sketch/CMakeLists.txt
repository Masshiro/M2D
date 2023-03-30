#   ./src/sketch/CMakeLists.txt
add_library(EMatrix EMatrix.hpp)
target_link_libraries(EMatrix  ${CONAN_LIBS} leader_zero)

add_library(EMatrixTC4bit EMatrixTC.hpp)
target_link_libraries(EMatrixTC4bit  ${CONAN_LIBS} leader_zero)

add_library(HyperLogLog HLL.hpp)
target_link_libraries(HyperLogLog  ${CONAN_LIBS} leader_zero)

add_library(PMatrix PMatrix.hpp)
target_link_libraries(PMatrix  ${CONAN_LIBS} leader_zero)
