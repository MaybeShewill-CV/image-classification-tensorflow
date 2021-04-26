cmake_minimum_required(VERSION 3.9)
project(workflow_monitor)

# complier options
set(CMAKE_C_STANDARD 99)
set(CMAKE_CXX_STANDARD 11)

# build options
option(WF_MONITOR_USE_CPP11 "Enable workflow monitor use c++11" ON)
option(WF_MONITOR_BUILD_TEST "Build test for workflow monitor" ON)

# target options
set(CMAKE_BUILD_TYPE Release)
IF(CMAKE_BUILD_TYPE MATCHES Debug)
    message("Build type debug")
ELSE()
    message("Build type release")
ENDIF(CMAKE_BUILD_TYPE MATCHES Debug)

message(STATUS ">>>>>>>>>>>>>")
message(STATUS "Monitor BUILD INFO:")
message(STATUS "\tSystem: ${CMAKE_SYSTEM_NAME}")
message(STATUS "\tProcessor: ${CMAKE_SYSTEM_PROCESSOR}")
message(STATUS "\tBuild Path: ${CMAKE_CURRENT_BINARY_DIR}")

if (NOT MSVC)
    if(WF_MONITOR_USE_CPP11)
        set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -std=gnu99")
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
    else()
        set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -std=gnu99")
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++0x")
    endif()
endif()

IF(CMAKE_BUILD_TYPE MATCHES Debug)
    if(MSVC)
        set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} /DEBUG")
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /DEBUG")
    else()
        set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -g")
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g")
    endif()
else()
    if (MSVC)
        set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} /O2")
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /O2")
    else()
        set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -O3")
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3")
        if(CMAKE_SYSTEM_NAME MATCHES "^Android")
            if(MNN_BUILD_FOR_ANDROID_COMMAND)
                set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} -s")
                set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -pie -fPIE -s")
                set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} -Wl,--gc-sections")
            endif()
        endif()
    endif()
ENDIF(CMAKE_BUILD_TYPE MATCHES Debug)

if (NOT MSVC)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fstrict-aliasing -ffunction-sections -fdata-sections -ffast-math -frtti -fexceptions ")
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -fstrict-aliasing -ffunction-sections -fdata-sections -ffast-math")
endif()

include_directories(${CMAKE_CURRENT_LIST_DIR}/include/
        ${CMAKE_CURRENT_LIST_DIR}/3rd_party/
        ${CMAKE_CURRENT_LIST_DIR}/3rd_party/flatbuffers/include
        ${CMAKE_CURRENT_LIST_DIR}/source/
        )

# utils
FILE(GLOB WF_MONITOR_UTILS_SRC ${CMAKE_CURRENT_LIST_DIR}/source/utils/*cpp)

# 3RD SOURCE
FILE(GLOB_RECURSE WF_MONITOR_3RD_SRC ${CMAKE_CURRENT_LIST_DIR}/3rd_party/*cpp)

find_package(OpenMP REQUIRED)
find_package(OpenSSL REQUIRED)
find_package(workflow REQUIRED CONFIG HINTS ..)
find_package(Boost REQUIRED)
find_library(LIBRT rt)

add_executable(
        workflow_monitor.out
        wf_net_training_monitor.cpp
        ${WF_MONITOR_UTILS_SRC}
        ${WF_MONITOR_3RD_SRC}
)
target_link_libraries(
        workflow_monitor.out
        ${CMAKE_CURRENT_LIST_DIR}/3rd_party/libs/libglog.so
        ${CMAKE_CURRENT_LIST_DIR}/3rd_party/libs/libboost_filesystem.so
        ${CMAKE_CURRENT_LIST_DIR}/3rd_party/libs/libboost_system.so
        ${CMAKE_CURRENT_LIST_DIR}/3rd_party/libs/libworkflow.so
        ${LIBRT}
        OpenSSL::Crypto
        OpenSSL::SSL
)

if (WF_MONITOR_BUILD_TEST)
    FILE(GLOB_RECURSE WF_MONITOR_TEST_SRC ${CMAKE_CURRENT_LIST_DIR}/3rd_party/*cc)
