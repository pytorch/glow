#ifdef GLOW_WITH_CPU

BB.newBackendSpecificInstr("CPUMaxSplat")
    .addOperand("Dest", OperandKind::Out)
    .addOperand("Src", OperandKind::In)
    .addMember(MemberType::Float, "SplatValue")
    .inplaceOperand({"Dest", "Src"})
    .dataParallel()
    .autoIRGen();

BB.newBackendSpecificInstr("CPUFullyConnected")
    .addOperand("Dest", OperandKind::Out)
    .addOperand("Src", OperandKind::In)
    .addOperand("Weights", OperandKind::In)
    .addOperand("Bias", OperandKind::In)
    .autoIRGen("FullyConnected")
    .autoVerify(VerifyKind::NoVerify);

BB.includeBackendSpecificVerification("CPUSpecificInstrsVerification.h");

#endif // GLOW_WITH_CPU
