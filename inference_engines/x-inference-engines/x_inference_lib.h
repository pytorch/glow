/**
 * Copyright (c) Glow Contributors. See CONTRIBUTORS file.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

/**
 * Contributed by Xperi Corporation on August 13, 2019
 */

#ifndef X_INFERENCE_LIB_H
#define X_INFERENCE_LIB_H

#include <stdint.h>
#include <stdlib.h>

#ifdef ENABLE_PERF_MONITORING
#include "x_perf_monitor.h"
#endif // ENABLE_PERF_MONITORING

/// Successful return
#define X_SUCCESS (0)
/// Failure
#define X_FAILURE (-1)

/// Inference function pointer type.
typedef void (*InferenceFunctionPtr_t)(uint8_t *, uint8_t *, uint8_t *);

/// The symbol table entry for Glow bundle symbols.
struct SymbolTableEntry {
  const char *name;
  uint64_t offset;
  uint64_t size;
  char kind;
};

/// The Glow bundle config structure.
struct BundleConfig {
  uint64_t constWeightVarsMemsize;
  uint64_t mutWeightVarsMemsize;
  uint64_t activationMemsize;
  uint64_t alignment;
  uint64_t numSymbols;
  const struct SymbolTableEntry *symbols;
};

/// The runtime data structure.
struct RuntimeData {
  /// Pointer to the location of constant weights in memory.
  uint8_t *constWeights;
  /// Pointer to the location of mutable weights in memory.
  uint8_t *mutWeights;
  /// Pointer to the location of activations in memory.
  uint8_t *activations;
  /// The inference function
  InferenceFunctionPtr_t inferenceFunc;
  /// Offset of the input tensor in memory
  size_t inputOffset;
  /// Offset of the output tensor in memory
  size_t outputOffset;

#ifdef ENABLE_PERF_MONITORING
  /// Performance statistics
  struct PerfStatistics ps;
  /// Should we monitor performance?
  int doPerfMonitoring;
#endif // ENABLE_PERF_MONITORING
};

/// Inference IO metadata
struct InferenceIO {
  /// Pointer to the input data
  void *input;
  /// Pointer to location of output buffer
  void *output;
  /// Should input memory be deallocated (yes if we allocated it, no if we were
  /// passed a pointer to a previously allocated region)
  int cleanupInput;
  /// Should output memory be deallocated (yes if we allocated it, no if we were
  /// passed a pointer to a previously allocated region)
  int cleanupOutput;
  /// Input data length
  size_t inLen;
  /// Output data length
  size_t outLen;
  /// Batch size
  size_t batchSize;
};

/// Network metadata
struct NetworkData {
  /// The Glow bundle config
  struct BundleConfig *bundleConfig;
  /// Name of the input tensor
  char *inputTensorName;
  /// Name of the output tensor
  char *outputTensorName;
  /// Name of the weights file
  char *weightsFileName;
  /// Inference function
  InferenceFunctionPtr_t inferenceFunction;
  /// Should we monitor performance?
  int doPerfMonitoring;
};

/// Initialize runtime data \p runtimeData given the network metadata \p
/// networkData. \returns X_SUCCESS on success, X_FAILURE on failure.
int initRuntimeData(const struct NetworkData *networkData,
                    struct RuntimeData *runtimeData);

/// Initialize inference IO metadata given pointers to the IO memory \p inMMap
/// and \p outMMap. \p inMMap may be null, then memory is allocated (and
/// deallocated when done). \p outMMap may be null, then memory is allocated
/// (and deallocated when done). If \p inMMap and \p outMMap are not null, IO
/// memory is not allocated/deallocated. returns X_SUCCESS on success, X_FAILURE
/// on failure.
int initIO(struct InferenceIO *iio, void *inMMap, void *outMMap);

/// Clean up the runtime data \p runtimeData, which ammounts to deallocating
/// allocated resources.
void cleanupRuntimeData(struct RuntimeData *runtimeData);

/// Clean up inference IO \p io, which ammounts to deallocating allocated
/// resources.
void cleanupIO(struct InferenceIO *io);

/// Run the inference given Inference IO \p iio and the runtime data \p
/// runtimeData.
void runInference(const struct InferenceIO *iio,
                  struct RuntimeData *runtimeData);

#endif // X_INFERENCE_LIB_H
