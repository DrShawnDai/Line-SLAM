cmake_minimum_required (VERSION 3.5)
project(line_slam)

set(CMAKE_BUILD_TYPE "Release")
set(CMAKE_CXX_FLAGS "-std=c++14")

find_package(OpenCV 4 REQUIRED)
find_package(Eigen3 3 REQUIRED)
find_package(Ceres REQUIRED)
find_package(Pangolin REQUIRED)

# message(STATUS "eigen include path: " ${EIGEN3_INCLUDE_DIR})
# message(STATUS "eigen libs: " ${EIGENS_LIBS})
# message(STATUS "opencv include path: " ${OpenCV_INCLUDE_DIRS})
# message(STATUS "opencv libs: " ${OpenCV_LIBS})
# message(STATUS "ceres libs: " ${CERES_LIBRARIES})
# message(STATUS "pangolin libs: " ${Pangolin_LIBRARIES})

include_directories(
    ${PROJECT_SOURCE_DIR}
    ${PROJECT_SOURCE_DIR}/include
    ${EIGEN3_INCLUDE_DIR}
    ${OpenCV_INCLUDE_DIRS}
    ${CERES_INCLUDE_DIRS}
    ${Pangolin_INCLUDE_DIRS}
)

set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/lib)

add_library(${PROJECT_NAME} SHARED
    src/Utility.cpp
    src/LBDextractor.cpp
    src/LBDmatcher.cpp
    src/KeyFrame.cpp
    src/MapPoint.cpp
    src/MapLine.cpp
    src/Map.cpp
    src/Viewer.cpp 
    src/MapDrawer.cpp
    src/Optimization.cpp
)

target_link_libraries(${PROJECT_NAME}
    ${OpenCV_LIBS}
    ${CERES_LIBRARIES}
    ${Pangolin_LIBRARIES}
    "/usr/local/opt/glew/lib/libGLEW.2.2.dylib"
)

# Build examples
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/examples/plucker)
add_executable(plucker examples/plucker/plucker.cpp)
target_link_libraries(plucker ${PROJECT_NAME})

# set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/examples/1bdmatch)
# add_executable(lbdmatch
# examples/lbdmatch/lbdmatch.cpp)
# target_link_libraries(lbdmatch ${PROJECT_NAME})

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/examples/optimize)
add_executable (optimize examples/optimize/optimize.cpp)
target_link_libraries(optimize ${PROJECT_NAME})