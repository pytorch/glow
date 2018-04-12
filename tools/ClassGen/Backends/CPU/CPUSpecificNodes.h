#ifdef GLOW_WITH_CPU

BB.newBackendSpecificNode("CPUMaxSplat")
    .addInput("Input")
    .addResult("Input.getType()")
    .addMember(MemberType::Float, "SplatValue")
    .setDocstring("A Max node with one splat input; CPU specific.");

BB.newNode("CPUConvDKKC8")
    .addInput("Input")
    .addInput("Filter")
    .addInput("Bias")
    .addMember(MemberType::SizeT, "Kernel")
    .addMember(MemberType::SizeT, "Stride")
    .addMember(MemberType::SizeT, "Pad")
    .addResultFromCtorArg()
    .setDocstring("This is a cpu-specific convolution implementation where the "
                  "filter is transposed to the shape [D/8, K, K, C, 8]");

BB.includeBackendSpecificVerification("CPUSpecificNodesVerification.h");

#endif // GLOW_WITH_CPU
