set(VERIF_FILENAME HabanaSpecificNodesVerification.h)
configure_file(${VERIF_FILENAME}
               ${GLOW_BINARY_DIR}/glow/${VERIF_FILENAME} COPYONLY)
