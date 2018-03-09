set(VERIF_FILENAME CPUSpecificNodesVerification.h)
configure_file(${VERIF_FILENAME} ${GLOW_BINARY_DIR}/${VERIF_FILENAME} COPYONLY)

set(VERIF_FILENAME CPUSpecificInstrsVerification.h)
configure_file(${VERIF_FILENAME} ${GLOW_BINARY_DIR}/${VERIF_FILENAME} COPYONLY)
