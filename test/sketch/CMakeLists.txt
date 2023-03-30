add_executable(test_leading_zero test_leading_zero.cpp)
target_link_libraries(test_leading_zero  ${CONAN_LIBS} leader_zero)

add_executable(test_hyperloglog test_hyperloglog.cpp)
target_link_libraries(test_hyperloglog  ${CONAN_LIBS} hyperloglog)

add_executable(test_xxhash test_xxhash.cpp)
target_link_libraries(test_xxhash  ${CONAN_LIBS} leader_zero)

add_executable(test_histogram test_histogram.cpp)
target_link_libraries(test_histogram  ${CONAN_LIBS} leader_zero)

add_executable(test_onvhll test_onvhll.cpp)
target_link_libraries(test_onvhll  ${CONAN_LIBS} leader_zero)

add_executable(test_vhll test_vhll.cpp)
target_link_libraries(test_vhll  ${CONAN_LIBS} leader_zero hyperloglog)

add_executable(test_CardinalityMap test_CardinalityMap.cpp)
target_link_libraries(test_CardinalityMap  ${CONAN_LIBS} leader_zero)

add_executable(test_tailcut test_tailcut.cpp)
target_link_libraries(test_tailcut  ${CONAN_LIBS} leader_zero)

add_executable(test_attack test_attack.cpp)
target_link_libraries(test_attack  ${CONAN_LIBS} leader_zero)

add_executable(test_multiflow test_multiflow.cpp)
target_link_libraries(test_multiflow  ${CONAN_LIBS} leader_zero)

add_executable(test_prefilter test_prefilter.cpp)
target_link_libraries(test_prefilter  ${CONAN_LIBS} leader_zero)

add_executable(true_distribution true_distribution.cpp)
target_link_libraries(true_distribution  ${CONAN_LIBS} leader_zero)

add_executable(test_minheap test_minheap.cpp)
target_link_libraries(test_minheap  ${CONAN_LIBS} leader_zero)

add_executable(test_fp_approximation test_fp_approximation.cpp)
target_link_libraries(test_fp_approximation  ${CONAN_LIBS})

add_executable(test_global_histogram test_global_histogram.cpp)
target_link_libraries(test_global_histogram  ${CONAN_LIBS}  leader_zero MurmurHash3)

add_executable(test_map_impl_filter test_map_impl_filter.cpp)
target_link_libraries(test_map_impl_filter ${CONAN_LIBS} leader_zero)