add_executable(test_univmon test_univmon.cpp)
target_link_libraries(test_univmon ${CONAN_LIBS} leader_zero)

add_executable(test_lus test_lus.cpp)
target_link_libraries(test_lus ${CONAN_LIBS} leader_zero)

add_executable(test_moment_estimator test_moment_estimator.cpp)
target_link_libraries(test_moment_estimator ${CONAN_LIBS} leader_zero)

add_executable(test_moment_estimator_pro test_moment_estimator_pro.cpp)
target_link_libraries(test_moment_estimator_pro ${CONAN_LIBS} leader_zero)

add_executable(test_moment_estimator_combined test_moment_estimator_combined.cpp)
target_link_libraries(test_moment_estimator_combined ${CONAN_LIBS} leader_zero)