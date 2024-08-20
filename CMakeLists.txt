cmake_minimum_required(VERSION 3.11)
project(verifyobject)

find_program(Docker_EXECUTABLE docker)
if(NOT Docker_EXECUTABLE)
    message(WARNING "Cannot find the docker-compose executable! Check package README for instructions on building the docker images manually.")
else()
    function(build_docker_image image_name)
      MESSAGE("COMMAND: ${Docker_EXECUTABLE} compose build ${image_name}")
      execute_process(COMMAND ${Docker_EXECUTABLE} compose build ${image_name}
          WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
          RESULT_VARIABLE CMD_ERROR_${image_name}
          OUTPUT_FILE CMD_OUTPUT_${image_name})
      MESSAGE(STATUS "[${image_name}] CMD_ERROR:" ${CMD_ERROR_${image_name}})
      MESSAGE(STATUS "[${image_name}] CMD_OUTPUT:" ${CMD_OUTPUT_${image_name}})
    endfunction()

    build_docker_image(verifyobject)
endif()

find_package(catkin REQUIRED)
catkin_package()

install(DIRECTORY config data launch src
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}
)

install(
  FILES
    Dockerfile
    ros_entrypoint_noetic.sh
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}
)

install(
  PROGRAMS
    src/start_container.sh
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)
