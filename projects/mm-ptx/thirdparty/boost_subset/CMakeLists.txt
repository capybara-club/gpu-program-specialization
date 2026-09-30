add_library(vendor_boost_preprocessor INTERFACE)

target_include_directories(vendor_boost_preprocessor SYSTEM INTERFACE
  ${CMAKE_CURRENT_LIST_DIR}   # contains "boost/"
)
