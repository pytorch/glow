set(VERIF_FILENAME OpenCLSpecificNodesVerification.h)
configure_file(${VERIF_FILENAME} ${GLOW_BINARY_DIR}/${VERIF_FILENAME} COPYONLY)

set(VERIF_FILENAME OpenCLSpecificInstrsVerification.h)
configure_file(${VERIF_FILENAME} ${GLOW_BINARY_DIR}/${VERIF_FILENAME} COPYONLY)
