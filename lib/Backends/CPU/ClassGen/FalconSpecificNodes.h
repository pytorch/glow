#ifdef GLOW_WITH_CPU
//#ifdef GLOW_WITH_FALCON

BB.newBackendSpecificNode("FalconMerged")
    .addInput("Input")
    .addInput("Filter")
    .addInput("Config")
    .addResult("Input.getType()")
    .setDocstring("FALCON merged layer for FPGA accelerator");

BB.includeBackendSpecificVerification("glow/FalconSpecificNodesVerification.h");

#endif // GLOW_WITH_FALCON