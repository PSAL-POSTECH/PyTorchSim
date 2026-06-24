CMAKE_MINIMUM_REQUIRED(VERSION 3.15)

# project setting
set(LIB_NAME "Simulator")

# set source and headers
set(CMAKE_CXX_STANDARD 23)
file(GLOB_RECURSE SRC_FILES
  "${CMAKE_SOURCE_DIR}/src/*.h"
  "${CMAKE_SOURCE_DIR}/src/*.cc"
)

# build
add_executable(${LIB_NAME} ${SRC_FILES})

# Export the executable's dynamic symbols (-rdynamic) so a dlopen'd trace
# producer .so resolves the togsim_* runtime callbacks back into this binary
# (P3 trace pipeline).
set_target_properties(${LIB_NAME} PROPERTIES ENABLE_EXPORTS ON)
