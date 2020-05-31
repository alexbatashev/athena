set(INT_TEST_SRC
  main.cpp)

add_athena_executable(
        TestIntegrationJITRunnable
        ${INT_TEST_SRC})

target_link_libraries(TestIntegrationJITRunnable PRIVATE
        athena
        AthenaDep::googletest)

add_test(NAME JITIntegrationTest COMMAND TestIntegrationJITRunnable)
