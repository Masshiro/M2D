add_executable(moment_estimation_accuracy moment_estimation_accuracy.cpp)
target_link_libraries(moment_estimation_accuracy ${CONAN_LIBS} leader_zero)

add_executable(multilayer_unbiasness_preload multilayer_unbiasness_preload.cpp)
target_link_libraries(multilayer_unbiasness_preload ${CONAN_LIBS} leader_zero)

add_executable(multilayer_unbiasness multilayer_unbiasness.cpp)
target_link_libraries(multilayer_unbiasness ${CONAN_LIBS} leader_zero)

add_executable(throughput_memory throughput_memory.cpp)
target_link_libraries(throughput_memory ${CONAN_LIBS} leader_zero)

add_executable(rerror_memory_cost rerror_memory_cost.cpp)
target_link_libraries(rerror_memory_cost ${CONAN_LIBS} leader_zero)

add_executable(level_cnt_memory_cost level_cnt_memory_cost.cpp)
target_link_libraries(level_cnt_memory_cost ${CONAN_LIBS} leader_zero)

add_executable(bias_memory bias_memory.cpp)
target_link_libraries(bias_memory ${CONAN_LIBS} leader_zero)

add_executable(bias_memory_pro bias_memory_pro.cpp)
target_link_libraries(bias_memory_pro ${CONAN_LIBS} leader_zero)

add_executable(sampling_card_ranges sampling_card_ranges.cpp)
target_link_libraries(sampling_card_ranges ${CONAN_LIBS} leader_zero)

add_executable(bias_memory_low_order bias_memory_low_order.cpp)
target_link_libraries(bias_memory_low_order ${CONAN_LIBS} leader_zero)