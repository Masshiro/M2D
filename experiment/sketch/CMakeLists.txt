add_executable(on_vhll_bias_variance on_vhll_bias_variance.cpp)
target_link_libraries(on_vhll_bias_variance  ${CONAN_LIBS} leader_zero)

add_executable(on_vllc_bias_variance on_vllc_bias_variance.cpp)
target_link_libraries(on_vllc_bias_variance  ${CONAN_LIBS} leader_zero)

add_executable(ton_vhll_bias_variance ton_vhll_bias_variance.cpp)
target_link_libraries(ton_vhll_bias_variance ${CONAN_LIBS} leader_zero)

add_executable(operation_statistics operation_statistics.cpp)
target_link_libraries(operation_statistics ${CONAN_LIBS} leader_zero)

add_executable(vhll_bias_variance vhll_bias_variance.cpp)
target_link_libraries(vhll_bias_variance ${CONAN_LIBS} leader_zero)

add_executable(topk_identification_accuracy topk_identification_accuracy.cpp)
target_link_libraries(topk_identification_accuracy ${CONAN_LIBS} leader_zero)