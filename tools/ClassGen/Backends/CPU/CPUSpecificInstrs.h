#ifdef GLOW_WITH_CPU

BB.newBackendSpecificInstr("CPUMaxSplat")
    .addOperand("Dest", OperandKind::Out)
    .addOperand("Src", OperandKind::In)
    .addMember(MemberType::Float, "SplatValue")
    .inplaceOperand({"Dest", "Src"})
    .autoIRGen();

BB.includeBackendSpecificVerification("CPUSpecificInstrsVerification.h");

#endif // GLOW_WITH_CPU
