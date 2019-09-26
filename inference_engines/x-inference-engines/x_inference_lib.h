/**
 * Copyright (c) 2017-present, Facebook, Inc.
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
  uint64_t const_weight_vars_memsize;
  uint64_t mut_weight_vars_memsize;
  uint64_t activation_memsize;
  uint64_t alignment;
  uint64_t num_symbols;
  const struct SymbolTableEntry *symbols;
};

/// The runtime data structure.
struct RuntimeData {
  /// Pointer to the location of constant weights in memory.
  uint8_t *const_weights;
  /// Pointer to the location of mutable weights in memory.
  uint8_t *mut_weights;
  /// Pointer to the location of activations in memory.
  uint8_t *activations;
  /// The inference function
  InferenceFunctionPtr_t inference_func;
  /// Offset of the input tensor in memory
  size_t input_offset;
  /// Offset of the output tensor in memory
  size_t output_offset;

#ifdef ENABLE_PERF_MONITORING
  /// Performance statistics
  struct PerfStatistics ps;
  /// Should we monitor performance?
  int do_perf_monitoring;
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
  int cleanup_input;
  /// Should output memory be deallocated (yes if we allocated it, no if we were
  /// passed a pointer to a previously allocated region)
  int cleanup_output;
  /// Input data length
  size_t in_len;
  /// Output data length
  size_t out_len;
  /// Batch size
  size_t batch_size;
};

/// Network metadata
struct NetworkData {
  /// The Glow bundle config
  struct BundleConfig *bundle_config;
  /// Name of the input tensor
  char *input_tensor_name;
  /// Name of the output tensor
  char *output_tensor_name;
  /// Name of the weights file
  char *weights_file_name;
  /// Inference function
  InferenceFunctionPtr_t inference_function;
  /// Should we monitor performance?
  int do_perf_monitoring;
};

/// Initialize runtime data \p runtime_data given the network metadata \p
/// network_data. \returns X_SUCCESS on success, X_FAILURE on failure.
int init_runtime_data(const struct NetworkData *network_data,
                      struct RuntimeData *runtime_data);

/// Initialize inference IO metadata given pointers to the IO memory \p in_mmap
/// and \p out_mmap. \p in_mmap may be null, then memory is allocated (and
/// deallocated when done). \p out_mmap may be null, then memory is allocated
/// (and deallocated when done). If \p in_mmap and \p out_mmap are not null, IO
/// memory is not allocated/deallocated. returns X_SUCCESS on success, X_FAILURE
/// on failure.
int init_io(struct InferenceIO *iio, void *in_mmap, void *out_mmap);

/// Clean up the runtime data \p runtime_data, which ammounts to deallocating
/// allocated resources.
void cleanup_runtime_data(struct RuntimeData *runtime_data);

/// Clean up inference IO \p io, which ammounts to deallocating allocated
/// resources.
void cleanup_io(struct InferenceIO *io);

/// Run the inference given Inference IO \p iio and the runtime data \p
/// runtime_data.
void run_inference(const struct InferenceIO *iio,
                   struct RuntimeData *runtime_data);

#endif // X_INFERENCE_LIB_H