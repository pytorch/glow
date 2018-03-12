#ifdef GLOW_WITH_CPU

BB.newBackendSpecificNode("CPUMaxSplat")
    .addInput("Input")
    .addResult("Input.getType()")
    .addMember(MemberType::Float, "SplatValue")
    .setDocstring("A Max node with one splat input; CPU specific.");

BB.includeBackendSpecificVerification("CPUSpecificNodesVerification.h");

#endif // GLOW_WITH_CPU
