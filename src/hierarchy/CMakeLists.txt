#   ./src/hierarchy/CMakeLists.txt
add_library(UnivMonPro UnivMonPro.hpp)
target_link_libraries(UnivMonPro ${CONAN_LIBS} leader_zero)

add_library(LUSPro LUSPro.hpp)
target_link_libraries(LUSPro ${CONAN_LIBS} leader_zero)

add_library(MomentEstimator MomentEstimator.hpp)
target_link_libraries(MomentEstimator ${CONAN_LIBS} leader_zero)

add_library(MomentEstimatorPro MomentEstimatorPro.hpp)
target_link_libraries(MomentEstimatorPro ${CONAN_LIBS} leader_zero)

add_library(MomentEstimatorCombined MomentEstimatorCombined.hpp)
target_link_libraries(MomentEstimatorCombined ${CONAN_LIBS} leader_zero)