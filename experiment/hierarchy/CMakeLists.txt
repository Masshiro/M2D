add_executable(bias_memory bias_memory.cpp)
target_link_libraries(bias_memory ${CONAN_LIBS} leader_zero)

add_executable(throughput throughput.cpp)
target_link_libraries(throughput ${CONAN_LIBS} leader_zero)