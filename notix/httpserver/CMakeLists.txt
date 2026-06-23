cmake_minimum_required(VERSION 3.10)
project(notix_httpserver LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

option(BUILD_HTTP_SERVER "Build the HTTP server executable" ON)
option(BUILD_LIBTESTS "Build Boost/jsoncpp smoke tests" ON)

set(INFERFLOW_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/../..")
set(INFERFLOW_THIRD_PARTY_DIR "${INFERFLOW_ROOT}/third_party")

set(LOCAL_BOOST_ROOT "${INFERFLOW_THIRD_PARTY_DIR}/boost_1_84_0")
set(LOCAL_JSONCPP_ROOT "${INFERFLOW_THIRD_PARTY_DIR}/jsoncpp-1.9.6")
set(LOCAL_JSONCPP_LIBRARY "${LOCAL_JSONCPP_ROOT}/build/lib/libjsoncpp.a")
set(LOCAL_MYSQLCLIENT_ROOT "${INFERFLOW_THIRD_PARTY_DIR}/mysqlclient")
set(LOCAL_MYSQLCLIENT_INCLUDE_DIR "${LOCAL_MYSQLCLIENT_ROOT}/usr/include")
set(LOCAL_MYSQLCLIENT_LIBRARY "${LOCAL_MYSQLCLIENT_ROOT}/usr/lib/x86_64-linux-gnu/libmysqlclient.so")

if(EXISTS "${LOCAL_BOOST_ROOT}/stage/lib/cmake/Boost-1.84.0/BoostConfig.cmake")
    list(PREPEND CMAKE_PREFIX_PATH "${LOCAL_BOOST_ROOT}/stage/lib/cmake")
    set(Boost_NO_SYSTEM_PATHS ON)
endif()

find_package(Boost REQUIRED COMPONENTS system)
find_package(Threads REQUIRED)

if(BUILD_HTTP_SERVER)
    add_subdirectory(
        "${CMAKE_CURRENT_SOURCE_DIR}/../../llama-engine"
        "${CMAKE_CURRENT_BINARY_DIR}/llama-engine-build"
    )
endif()

if(EXISTS "${LOCAL_JSONCPP_ROOT}/include/jsoncpp/json/json.h" AND EXISTS "${LOCAL_JSONCPP_LIBRARY}")
    set(JSONCPP_FOUND TRUE)
    set(JSONCPP_INCLUDE_DIRS "${LOCAL_JSONCPP_ROOT}/include")
    set(JSONCPP_LIBRARIES "${LOCAL_JSONCPP_LIBRARY}")
else()
    find_package(PkgConfig QUIET)
    if(PkgConfig_FOUND)
        pkg_check_modules(JSONCPP QUIET jsoncpp)
    endif()
endif()

if(NOT JSONCPP_FOUND)
    find_path(JSONCPP_INCLUDE_DIR
        NAMES json/json.h jsoncpp/json/json.h
        PATH_SUFFIXES jsoncpp
    )
    find_library(JSONCPP_LIBRARY NAMES jsoncpp)

    if(NOT JSONCPP_LIBRARY)
        message(FATAL_ERROR "jsoncpp library not found. Please install libjsoncpp-dev.")
    endif()

    set(JSONCPP_INCLUDE_DIRS ${JSONCPP_INCLUDE_DIR})
    set(JSONCPP_LIBRARIES ${JSONCPP_LIBRARY})
endif()

if(EXISTS "${LOCAL_MYSQLCLIENT_INCLUDE_DIR}/mysql/mysql.h" AND EXISTS "${LOCAL_MYSQLCLIENT_LIBRARY}")
    set(MYSQL_FOUND TRUE)
    set(MYSQL_INCLUDE_DIRS "${LOCAL_MYSQLCLIENT_INCLUDE_DIR}")
    set(MYSQL_LIBRARIES "${LOCAL_MYSQLCLIENT_LIBRARY}")
else()
    find_package(PkgConfig QUIET)
    if(PkgConfig_FOUND)
        pkg_check_modules(MYSQL QUIET mysqlclient)
    endif()
endif()

if(NOT MYSQL_FOUND)
    find_path(MYSQL_INCLUDE_DIR NAMES mysql/mysql.h)
    find_library(MYSQL_LIBRARY NAMES mysqlclient)

    if(NOT MYSQL_LIBRARY OR NOT MYSQL_INCLUDE_DIR)
        if(BUILD_HTTP_SERVER)
            message(FATAL_ERROR "mysqlclient headers or library not found. Please install libmysqlclient-dev.")
        endif()
        set(MYSQL_INCLUDE_DIRS "")
        set(MYSQL_LIBRARIES "")
    else()
        set(MYSQL_INCLUDE_DIRS ${MYSQL_INCLUDE_DIR})
        set(MYSQL_LIBRARIES ${MYSQL_LIBRARY})
    endif()
endif()

file(GLOB_RECURSE HTTPSERVER_SOURCES CONFIGURE_DEPENDS
    "${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp"
)

if(NOT HTTPSERVER_SOURCES)
    message(FATAL_ERROR "No source files found under ${CMAKE_CURRENT_SOURCE_DIR}/src")
endif()

if(BUILD_LIBTESTS)
    add_executable(test_boost "${INFERFLOW_ROOT}/libtest/testboost.cpp")
    target_include_directories(test_boost PRIVATE "${LOCAL_BOOST_ROOT}")
    target_link_libraries(test_boost PRIVATE Boost::system Threads::Threads)

    add_executable(test_json "${INFERFLOW_ROOT}/libtest/testjson.cpp")
    target_include_directories(test_json PRIVATE ${JSONCPP_INCLUDE_DIRS})
    target_link_libraries(test_json PRIVATE ${JSONCPP_LIBRARIES})
endif()

if(BUILD_HTTP_SERVER)
    add_executable(http_server ${HTTPSERVER_SOURCES})

    target_include_directories(http_server PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/inc
        ${CMAKE_CURRENT_SOURCE_DIR}/inc/http
        ${CMAKE_CURRENT_SOURCE_DIR}/inc/middleware
        ${CMAKE_CURRENT_SOURCE_DIR}/inc/router
        ${JSONCPP_INCLUDE_DIRS}
        ${MYSQL_INCLUDE_DIRS}
    )

    target_compile_definitions(http_server PRIVATE
        INFERFLOW_DEFAULT_MODEL_PATH="${CMAKE_CURRENT_SOURCE_DIR}/../../llama-engine/test/tinyllama.bin"
        INFERFLOW_DEFAULT_TOKENIZER_PATH="${CMAKE_CURRENT_SOURCE_DIR}/../../../models/tokenizer.model"
    )

    target_link_libraries(http_server PRIVATE
        llama_engine
        Boost::system
        Threads::Threads
        ${JSONCPP_LIBRARIES}
        ${MYSQL_LIBRARIES}
    )

    # 运行目录位于 build 时，自动复制静态资源
    add_custom_command(TARGET http_server POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E make_directory "$<TARGET_FILE_DIR:http_server>/static"
        COMMAND ${CMAKE_COMMAND} -E copy_directory
                "${CMAKE_CURRENT_SOURCE_DIR}/static"
                "$<TARGET_FILE_DIR:http_server>/static"
        COMMENT "Copy static resources"
    )
endif()

message(STATUS "Project source dir: ${CMAKE_CURRENT_SOURCE_DIR}")
message(STATUS "Sources: ${HTTPSERVER_SOURCES}")
