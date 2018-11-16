cmake_minimum_required(VERSION 2.8 FATAL_ERROR)

project(convert_pcd_ascii_binary)

find_package(PCL 1.2 REQUIRED)

include_directories(${PCL_INCLUDE_DIRS})
link_directories(${PCL_LIBRARY_DIRS})
add_definitions(${PCL_DEFINITIONS})

add_executable (convert_pcd_ascii_binary convert_pcd_ascii_binary.cpp)
target_link_libraries (convert_pcd_ascii_binary ${PCL_LIBRARIES})
