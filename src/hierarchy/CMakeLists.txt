#   ./src/hierarchy/CMakeLists.txt
add_library(MomentEstimator MomentEstimator.hpp)
target_link_libraries(MomentEstimator ${CONAN_LIBS} leader_zero)


add_library(MomentEstimatorCombined MomentEstimatorCombined.hpp)
target_link_libraries(MomentEstimatorCombined ${CONAN_LIBS} leader_zero)