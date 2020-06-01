set(INT_TEST_SRC 
  core.cpp
  ir.cpp)

add_athena_executable(
        TestIntegrationJITRunnable
        ${INT_TEST_SRC})

find_package(Threads)
target_link_libraries(TestIntegrationJITRunnable PRIVATE
        athena
        AthenaDep::googletest
        Threads::Threads)

add_test(NAME JITIntegrationTest COMMAND TestIntegrationJITRunnable)
