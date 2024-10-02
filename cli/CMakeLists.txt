if (NOT IS_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/../third_party/cxxopts/include")
  message(FATAL_ERROR "The client dependency repository (cxxopts) is missing! "
    "You probably didn't clone the project with --recursive. You can include it "
    "by calling \"git submodule update --init --recursive\"")
endif()

add_executable(translator
  translator.cc
  )
target_include_directories(translator
  PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../third_party/cxxopts/include
  )
if(WITH_CUDA)
  if(GPU_RUNTIME STREQUAL "HIP")
    find_package(hipblas REQUIRED)
    find_package(miopen REQUIRED)
    target_link_libraries(translator
            PRIVATE ${PROJECT_NAME} MIOpen #roc::hipblas
    )
  else()
    target_link_libraries(translator
      PRIVATE ${PROJECT_NAME}
  )
  endif()

else()
  target_link_libraries(translator
    PRIVATE ${PROJECT_NAME}
  )
endif()

set_target_properties(translator PROPERTIES OUTPUT_NAME ct2-translator)

install(
  TARGETS translator
  DESTINATION ${CMAKE_INSTALL_BINDIR}
  )
