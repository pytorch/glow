#ifdef GLOW_WITH_CPU

BB.newBackendSpecificInstr("CPUMaxSplat")
    .addOperand("Dest", OperandKind::Out)
    .addOperand("Src", OperandKind::In)
    .addMember(MemberType::Float, "SplatValue")
    .inplaceOperand({"Dest", "Src"})
    .dataParallel()
    .autoIRGen();

BB.newBackendSpecificInstr("CPUConvDKKC8")
    .addOperand("Dest", OperandKind::Out)
    .addOperand("Src", OperandKind::In)
    .addOperand("Filter", OperandKind::In)
    .addOperand("Bias", OperandKind::In)
    .addMember(MemberType::SizeT, "Kernel")
    .addMember(MemberType::SizeT, "Stride")
    .addMember(MemberType::SizeT, "Pad")
    .autoIRGen()
    .autoVerify(VerifyKind::SameElementType, {"Dest", "Src", "Filter", "Bias"});

BB.includeBackendSpecificVerification("CPUSpecificInstrsVerification.h");

#endif // GLOW_WITH_CPU
