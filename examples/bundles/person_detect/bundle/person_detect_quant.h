// Bundle API auto-generated header file. Do not edit!
// Glow Tools version: 2023-04-04 (565a0beeb) ()

#ifndef _GLOW_BUNDLE_PERSON_DETECT_QUANT_H
#define _GLOW_BUNDLE_PERSON_DETECT_QUANT_H

#include <stdint.h>

// ---------------------------------------------------------------
//                       Common definitions
// ---------------------------------------------------------------
#ifndef _GLOW_BUNDLE_COMMON_DEFS
#define _GLOW_BUNDLE_COMMON_DEFS

// Glow bundle error code for correct execution.
#define GLOW_SUCCESS 0

// Memory alignment definition with given alignment size
// for static allocation of memory.
#define GLOW_MEM_ALIGN(size)  __attribute__((aligned(size)))

// Macro function to get the absolute address of a
// placeholder using the base address of the mutable
// weight buffer and placeholder offset definition.
#define GLOW_GET_ADDR(mutableBaseAddr, placeholderOff)  (((uint8_t*)(mutableBaseAddr)) + placeholderOff)

#endif

// ---------------------------------------------------------------
//                          Bundle API
// ---------------------------------------------------------------
// Model name: "person_detect_quant"
// Total data size: 308160 (bytes)
// Activations allocation efficiency: 1.0000
// Placeholders:
//
//   Name: "input"
//   Type: i8[S:0.007843138 O:-1][-0.996,1.004]<1 x 96 x 96 x 1>
//   Size: 9216 (elements)
//   Size: 9216 (bytes)
//   Offset: 0 (bytes)
//
//   Name: "MobilenetV1_Predictions_Reshape_1"
//   Type: i8[S:0.003906250 O:-128][0.000,0.996]<1 x 2>
//   Size: 2 (elements)
//   Size: 2 (bytes)
//   Offset: 9216 (bytes)
//
// NOTE: Placeholders are allocated within the "mutableWeight"
// buffer and are identified using an offset relative to base.
// ---------------------------------------------------------------
#ifdef __cplusplus
extern "C" {
#endif

// Placeholder address offsets within mutable buffer (bytes).
#define PERSON_DETECT_QUANT_input                              0
#define PERSON_DETECT_QUANT_MobilenetV1_Predictions_Reshape_1  9216

// Memory sizes (bytes).
#define PERSON_DETECT_QUANT_CONSTANT_MEM_SIZE     243584
#define PERSON_DETECT_QUANT_MUTABLE_MEM_SIZE      9280
#define PERSON_DETECT_QUANT_ACTIVATIONS_MEM_SIZE  55296

// Memory alignment (bytes).
#define PERSON_DETECT_QUANT_MEM_ALIGN  64

// Bundle entry point (inference function). Returns 0
// for correct execution or some error code otherwise.
int person_detect_quant(uint8_t *constantWeight, uint8_t *mutableWeight, uint8_t *activations);

#ifdef __cplusplus
}
#endif
#endif
