include(${CMAKE_CURRENT_LIST_DIR}/modules/BinToInc.cmake)
message(STATUS "Generating: ${INC_FILE} from ${BIN_FILE}")
BinToInc(BIN_FILE "${BIN_FILE}" INC_FILE "${INC_FILE}")
