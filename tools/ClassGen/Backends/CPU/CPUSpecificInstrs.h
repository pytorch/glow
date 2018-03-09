#ifdef GLOW_WITH_CPU

BB.newBackendSpecificInstr("CPUMaxZero")
    .addOperand("Dest", OperandKind::Out)
    .addOperand("Src", OperandKind::In)
    .inplaceOperand({"Dest", "Src"})
    .autoIRGen();

BB.includeBackendSpecificVerification("CPUSpecificInstrsVerification.h");

#endif // GLOW_WITH_CPU
