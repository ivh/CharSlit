cmake_minimum_required(VERSION 3.15...3.27)

project(charslit LANGUAGES C CXX)

# Find Python
find_package(Python 3.11 REQUIRED COMPONENTS Interpreter Development.Module)

# Find nanobind
execute_process(
  COMMAND "${Python_EXECUTABLE}" -m nanobind --cmake_dir
  OUTPUT_STRIP_TRAILING_WHITESPACE
  OUTPUT_VARIABLE NB_DIR
  RESULT_VARIABLE NB_RESULT
)

if(NOT NB_RESULT EQUAL 0)
  message(FATAL_ERROR "nanobind not found. Install it with: pip install nanobind")
endif()

list(APPEND CMAKE_PREFIX_PATH "${NB_DIR}")
find_package(nanobind CONFIG REQUIRED)

# Create nanobind extension module
nanobind_add_module(
  charslit
  STABLE_ABI
  NB_STATIC
  slitdec/slitdec_wrapper.cpp
  slitdec/slitdec.c
)

# Explicitly set language for C file
set_source_files_properties(slitdec/slitdec.c PROPERTIES LANGUAGE C)

target_include_directories(charslit PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/slitdec
)

# Install the extension module
install(TARGETS charslit LIBRARY DESTINATION .)
