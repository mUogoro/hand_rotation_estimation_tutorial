cmake_minimum_required (VERSION 3.1)

project(HandRotation)

find_package(RGBDGrabber REQUIRED)
find_package(Padenti REQUIRED)

include_directories(${CMAKE_SOURCE_DIR}/src ${CMAKE_SOURCE_DIR}/src/mean_shift ${CMAKE_SOURCE_DIR}/src/particle_filter
                    ${RGBDGRABBER_INCLUDE_DIRS} ${PADENTI_INCLUDE_DIRS})
link_directories(${RGBDGRABBER_LIBRARY_DIRS} ${PADENTI_LIBRARY_DIRS})

add_executable(retrofit_rtree ${CMAKE_SOURCE_DIR}/src/mean_shift/rchord_meanshift.cpp
               ${CMAKE_SOURCE_DIR}/test/retrofit_rtree.cpp)
target_link_libraries(retrofit_rtree ${PADENTI_LIBRARIES})

add_executable(live_test ${CMAKE_SOURCE_DIR}/src/mean_shift/rchord_meanshift.cpp
			 ${CMAKE_SOURCE_DIR}/test/live_test.cpp)
target_link_libraries(live_test ${RGBDGRABBER_LIBRARIES} ${PADENTI_LIBRARIES} opencv_video)
