include(${CMAKE_CURRENT_LIST_DIR}/WrapString.cmake)

# Function to embed contents of a file as byte array in C/C++ include file(.inc). The include file
# will contain a byte array and integer variable holding the size of the array.
# Parameters
#   BIN_FILE    - The path of source file whose contents will be embedded in the include file.
#   INC_FILE    - The path of include file.
#   STRIP_COMMA - If specified then removes the trailing comma 
#   APPEND      - If specified appends to the include file instead of overwriting it
#
# Usage:
#   BinToInc(BIN_FILE "Logo.png" INC_FILE "Logo.inc")
#   BinToInc(BIN_FILE "Logo.png" INC_FILE "Logo.inc" STRIP_COMMA) Removes trailing comma
function(BinToInc)
    set(options APPEND STRIP_COMMA)
    set(oneValueArgs BIN_FILE INC_FILE)
    cmake_parse_arguments(B2I "${options}" "${oneValueArgs}" "" ${ARGN})

    # reads source file contents as hex string
    file(READ ${B2I_BIN_FILE} hexString HEX)

    # wraps the hex string into multiple lines at column 32(i.e. 16 bytes per line)
    WrapString(VARIABLE hexString AT_COLUMN 32)

    # adds '0x' prefix and comma suffix before and after every byte respectively
    string(REGEX REPLACE "([0-9a-f][0-9a-f])" "0x\\1, " arrayValues ${hexString})

    if (B2I_STRIP_COMMA)
        # removes trailing comma
        string(REGEX REPLACE ", $" "" arrayValues ${arrayValues})
    endif()

    if(B2I_APPEND)
        file(APPEND ${B2I_INC_FILE} "${arrayValues}")
    else()
        file(WRITE ${B2I_INC_FILE} "${arrayValues}")
    endif()
endfunction()
