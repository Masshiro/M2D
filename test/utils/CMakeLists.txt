add_executable(test_truthlist test_truthlist.cpp)
target_link_libraries(test_truthlist ${CONAN_LIBS} leader_zero)

add_executable(test_leading_zeros test_leading_zeros.cpp)
target_link_libraries(test_leading_zeros ${CONAN_LIBS})