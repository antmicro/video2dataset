cmake_minimum_required(VERSION 2.8)
project(alov-dataset-creator)

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_SOURCE_DIR}/third-party/GOTURN/cmake/Modules/")

find_package(TinyXML REQUIRED)
find_package(OpenCV REQUIRED )
find_package(CUDA REQUIRED)
find_package(Caffe REQUIRED)
find_package(Boost COMPONENTS system filesystem regex REQUIRED)

add_definitions(${OpenCV_DEFINITIONS})
include_directories(${CUDA_INCLUDE_DIRS})
include_directories(${Caffe_INCLUDE_DIRS})
add_definitions(${Caffe_DEFINITIONS})

set(GLOG_LIB glog)

message("OpenCV version is ${OpenCV_VERSION}")
message("OpenCV_DEFINITIONS is ${OpenCV_DEFINITIONS}")
message("CUDA_INCLUDE_DIRS is ${CUDA_INCLUDE_DIRS}")
message("Caffe_DEFINITIONS is ${Caffe_DEFINITIONS}")
message("Caffe_DIR is ${Caffe_DIR}")
message("Caffe_INCLUDE_DIRS is ${Caffe_INCLUDE_DIRS}")

set (CMAKE_CXX_STANDARD 11)

# Add src to include directories.
include_directories(src)
include_directories(third-party/include)
include_directories(third-party/GOTURN/src)
include_directories(third-party/GOTURN/src/native)

add_library (GOTURN
third-party/GOTURN/src/helper/bounding_box.cpp
third-party/GOTURN/src/train/example_generator.cpp
third-party/GOTURN/src/helper/helper.cpp
third-party/GOTURN/src/helper/high_res_timer.cpp
third-party/GOTURN/src/helper/image_proc.cpp
third-party/GOTURN/src/loader/loader_alov.cpp
third-party/GOTURN/src/loader/loader_imagenet_det.cpp
third-party/GOTURN/src/network/regressor.cpp
third-party/GOTURN/src/network/regressor_base.cpp
third-party/GOTURN/src/network/regressor_train.cpp
third-party/GOTURN/src/network/regressor_train_base.cpp
third-party/GOTURN/src/tracker/tracker.cpp
third-party/GOTURN/src/tracker/tracker_manager.cpp
third-party/GOTURN/src/train/tracker_trainer.cpp
third-party/GOTURN/src/loader/video.cpp
)
target_link_libraries(GOTURN ${OpenCV_LIBS} ${Boost_LIBRARIES} ${Caffe_LIBRARIES} ${GLOG_LIB})

add_executable(${PROJECT_NAME}
    src/alov-dataset-creator.cpp
)
target_link_libraries(${PROJECT_NAME}
    ${OpenCV_LIBS} ${OpenCV_LIBS} GOTURN)
