#ifdef GLOW_WITH_CPU

BB.newBackendSpecificInstr("FalconMerged")
    .addOperand("Dest", OperandKind::Out)
    .addOperand("Src", OperandKind::In)
    .addOperand("Filter", OperandKind::In)
    .addOperand("Config", OperandKind::In)
    .autoIRGen();

BB.includeBackendSpecificVerification("glow/FalconSpecificInstrsVerification.h");

#endif // GLOW_WITH_CPU