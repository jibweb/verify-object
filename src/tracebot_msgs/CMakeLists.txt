cmake_minimum_required(VERSION 3.0.2)
project(tracebot_msgs)

find_package(catkin REQUIRED COMPONENTS
  actionlib
  actionlib_msgs
  geometry_msgs
  message_generation
  sensor_msgs
  std_msgs
  trajectory_msgs
  vision_msgs
)

add_message_files(
  FILES
     DTQueryKnowledgeQuery.msg
     DTQueryKnowledgeAnswer.msg
     DTRobotParameterQuery.msg
     DTRobotParameterAnswer.msg
     DTVisualMatchAnswer.msg
     SerializedMsg.msg
     GraspType.msg
     CloseType.msg
     ReleaseType.msg
)
add_action_files(
  DIRECTORY action
  FILES
    ComputeTrajectory.action
    Dummy.action
    DTAssertNewObject.action
    DTQueryKnowledge.action
    DTRequestColorImage.action
    DTRequestComponentRegion.action
    DTRequestObjectComponentPose.action
    DTRequestObjectPose.action
    DTRequestRegion.action
    DTRequestRegionColorImage.action
    DTRequestRobotParameter.action
    DTUpdateObject.action
    DTVisualExpectationMatch.action
    ExecuteGrasp.action
    ExecuteTrajectory.action
    InsertCanister.action
    LiftObject.action
    LocateObject.action
    VerifyObject.action
    PreshapeGrasp.action
    ReleaseGrasp.action
)

generate_messages(
  DEPENDENCIES
    actionlib_msgs
    geometry_msgs
    sensor_msgs
    std_msgs
    trajectory_msgs
    vision_msgs
)

catkin_package(
  CATKIN_DEPENDS
    actionlib
    actionlib_msgs
    geometry_msgs
    message_runtime
    sensor_msgs
    std_msgs
    vision_msgs
)
