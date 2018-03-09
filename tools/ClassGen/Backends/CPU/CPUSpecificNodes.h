#ifdef GLOW_WITH_CPU

BB.newBackendSpecificNode("CPUMaxZero")
    .addInput("Input")
    .addResult("Input.getType()")
    .setDocstring("A Max node with one ZeroNode input; CPU specific.");

BB.includeBackendSpecificVerification("CPUSpecificNodesVerification.h");

#endif // GLOW_WITH_CPU
