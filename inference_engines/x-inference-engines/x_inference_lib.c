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

// Needed to expose (and silence warnings about implicit declaration of)
// posix_memalign().
#define _POSIX_C_SOURCE (200112L)

#include <errno.h>
#include <fcntl.h>
#include <memory.h>
#include <stdbool.h>
#include <stdio.h>
#include <string.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>

#ifdef ENABLE_PERF_MONITORING
#include <x_perf_monitor.h>
#endif // ENABLE_PERF_MONITORING

#include "x_inference_lib.h"

#ifdef ENABLE_PERF_MONITORING
static struct PerfData global_pd;
#endif // ENABLE_PERF_MONITORING

static uint8_t *init_constant_weights(const char *wfname,
                                      const struct BundleConfig *bundle_config);
static uint8_t *init_mutable_weights(const struct BundleConfig *bundle_config);
static uint8_t *init_activations(const struct BundleConfig *bundle_config);
static int get_io_offsets(const struct NetworkData *network_data,
                          size_t *in_offset, size_t *out_offset);

static void *malloc_aligned(uint64_t alignment, size_t size);

int init_runtime_data(const struct NetworkData *network_data,
                      struct RuntimeData *runtime_data) {
  uint8_t *result = NULL;
  int retval = X_FAILURE;

  (void)memset(runtime_data, 0x0, sizeof(struct RuntimeData));

// If performance monitoring is enabled, make sure to correctly clear/initialize
// the corresponding runtime data.
#ifdef ENABLE_PERF_MONITORING
  runtime_data->do_perf_monitoring = network_data->do_perf_monitoring;
  if (runtime_data->do_perf_monitoring) {
    (void)memset(&(runtime_data->ps), 0x0, sizeof(struct PerfStatistics));
    (void)memset(&global_pd, 0x0, sizeof(struct PerfData));

    // Initialize performance monitor.
    retval = init_perf_monitoring(&global_pd);
    if (retval == -1) {
      perror("ERROR: unable to initialize perf event");
      return X_FAILURE;
    }
  }
#endif // ENABLE_PERF_MONITORING

  result = init_constant_weights(network_data->weights_file_name,
                                 network_data->bundle_config);
  if (result != NULL) {
    runtime_data->const_weights = result;

// If performance monitoring is enabled, report total weights size as part of
// the monitoring.
#ifdef ENABLE_PERF_MONITORING
    if (runtime_data->do_perf_monitoring) {
      global_pd.ps.const_weights_size =
          network_data->bundle_config->const_weight_vars_memsize;
    }
#endif // ENABLE_PERF_MONITORING
       // Cascaded initialization of mutable weights->activations->get offsets.
    result = init_mutable_weights(network_data->bundle_config);
    if (result != NULL) {
      runtime_data->mut_weights = result;
      result = init_activations(network_data->bundle_config);
      if (result != NULL) {
        runtime_data->activations = result;
        runtime_data->inference_func = network_data->inference_function;

        // Where are the input and output tensors located?
        retval = get_io_offsets(network_data, &(runtime_data->input_offset),
                                &(runtime_data->output_offset));
      }
    }
  }

  return retval;
}

int init_io(struct InferenceIO *iio, void *in_mmap, void *out_mmap) {
  int retval = X_FAILURE;

  // If memory-mapped input is passed, we don't need to allocate memory
  // for input. Otherwise, allocate and require cleanup later.
  if (in_mmap != NULL) {
    iio->input = in_mmap;
    iio->cleanup_input = 0;
  } else {
    iio->input = malloc(iio->batch_size * iio->in_len);
    iio->cleanup_input = 1;
  }

  // If memory-mapped output is passed, we don't need to allocate memory
  // for output. Otherwise, allocate and require cleanup later. This should only
  // be done if we have non-null input.
  if (iio->input != NULL) {
    if (out_mmap != NULL) {
      iio->output = out_mmap;
      iio->cleanup_output = 0;
    } else {
      iio->output = malloc(iio->batch_size * iio->out_len);
      iio->cleanup_output = 1;
    }

    if (iio->output != NULL) {
      retval = X_SUCCESS;
    }
  }

  // This will return X_FAILURE. Success is returned if and only if both input
  // and output are not null (either memory mapped, or successfully allocated)
  return retval;
}

void cleanup_runtime_data(struct RuntimeData *runtime_data) {
  free(runtime_data->activations);
  free(runtime_data->const_weights);
  free(runtime_data->mut_weights);

  runtime_data->activations = NULL;
  runtime_data->const_weights = NULL;
  runtime_data->mut_weights = NULL;

#ifdef ENABLE_PERF_MONITORING
  if (runtime_data->do_perf_monitoring) {
    (void)stop_perf_monitoring(&global_pd);
  }
#endif // ENABLE_PERF_MONITORING
}

void cleanup_io(struct InferenceIO *iio) {
  if (iio->cleanup_input) {
    free(iio->input);
  }
  if (iio->cleanup_output) {
    free(iio->output);
  }

  iio->input = NULL;
  iio->output = NULL;
}

void run_inference(const struct InferenceIO *iio,
                   struct RuntimeData *runtime_data) {
  size_t batch_counter;
  size_t current_input_offset;
  size_t current_output_offset;

  if (iio->batch_size == 0) {
    return;
  }

  current_input_offset = 0;
  current_output_offset = 0;

#ifdef ENABLE_PERF_MONITORING
  if (runtime_data->do_perf_monitoring) {
    global_pd.ps.num_cases = iio->batch_size;
  }
#endif // ENABLE_PERF_MONITORING

  for (batch_counter = 0; batch_counter < iio->batch_size; ++batch_counter) {
    // Store the next input into the correct mutable weights location.
    (void)memcpy(runtime_data->mut_weights + runtime_data->input_offset,
                 iio->input + current_input_offset, iio->in_len);

#ifdef ENABLE_PERF_MONITORING
    if (runtime_data->do_perf_monitoring) {
      (void)resume_perf_monitoring(&global_pd);
    }
#endif // ENABLE_PERF_MONITORING
       // Call our inference function on the input data. Pass in locations of
       // constant weights, mutable weights (inputs/outputs), and activations.
    (runtime_data->inference_func)(runtime_data->const_weights,
                                   runtime_data->mut_weights,
                                   runtime_data->activations);
#ifdef ENABLE_PERF_MONITORING
    if (runtime_data->do_perf_monitoring) {
      (void)pause_perf_monitoring(&global_pd);
      (void)read_perf_statistics(&global_pd);
      (void)memcpy(&(runtime_data->ps), &(global_pd.ps),
                   sizeof(struct PerfStatistics));
    }
#endif // ENABLE_PERF_MONITORING

    // Store the current output into the correct output location.
    (void)memcpy(iio->output + current_output_offset,
                 runtime_data->mut_weights + runtime_data->output_offset,
                 iio->out_len);

    // Advance the input/output pointers to the next position of the next input
    // in the batch, and the corresponding output.
    current_input_offset += iio->in_len;
    current_output_offset += iio->out_len;
  }
}

/// Initialize constant weights with the data from the weights file.
/// \p wfname - Constant weights file name.
/// \p bundle_config - The bundle config structure
/// \returns Pointer to the constant weights on success, NULL on failure.
uint8_t *init_constant_weights(const char *wfname,
                               const struct BundleConfig *bundle_config) {
  size_t size = 0;
  int fd = 0;
  off_t file_offset = 0;
  uint8_t *retval = NULL;
  uint8_t *buffer = NULL;
  int bytes_read = 0;
  size_t bytes_total = 0;

  fd = open(wfname, O_RDONLY);
  if (fd == -1) {
    perror("Error processing weights file");
    return NULL;
  }

  file_offset = lseek(fd, 0, SEEK_END);
  if (file_offset == -1) {
    perror("Error processing weights file");
  } else {
    // Make sure the weights file is of expected size!
    size = (size_t)(file_offset);
    if (size != bundle_config->const_weight_vars_memsize) {
      fprintf(stderr,
              "Unexpected file size (%zd) does not match expected (%llu)\n",
              size, bundle_config->const_weight_vars_memsize);

      (void)close(fd);
      return NULL;
    }

    file_offset = lseek(fd, 0, SEEK_SET);
    if (file_offset == -1) {
      perror("Error processing weights file");

      (void)close(fd);
      return NULL;
    }

    retval = malloc_aligned(bundle_config->alignment, size);
    buffer = retval;
    // Read in the weights.
    if (retval != NULL) {
      while (bytes_total < size) {
        bytes_read = read(fd, buffer, size - bytes_total);
        bytes_total += bytes_read;

        if (bytes_read <= 0) {
          if (bytes_read == -1) {
            perror("Error reading weights file");
          } else if (bytes_read == 0) {
            fprintf(stderr,
                    "Error reading weights file: EOF reached too early\n");
          }

          free(retval);
          retval = NULL;
          break;
        }

        buffer += bytes_read;
      }
    } else
      perror("Error allocating memory for weights");
  }

  (void)close(fd);
  return retval;
}

/// Initialize mutable weights -- simply allocates the memory.
/// \returns Valid pointer on success, NULL on failure.
uint8_t *init_mutable_weights(const struct BundleConfig *bundle_config) {
  return malloc_aligned(bundle_config->alignment,
                        bundle_config->mut_weight_vars_memsize);
}

/// Initialize activations -- simply allocates the memory.
/// \returns Valid pointer on success, NULL on failure.
uint8_t *init_activations(const struct BundleConfig *bundle_config) {
  return malloc_aligned(bundle_config->alignment,
                        bundle_config->activation_memsize);
}

/// Performs aligned memory allocation.
/// \p alignment - alignment requirement
/// \p size - size of memory block to allocate
/// \returns Valid pointer on success, NULL on failure.
void *malloc_aligned(uint64_t alignment, size_t size) {
  int result = 0;
  void *retval = NULL;

  result = posix_memalign(&retval, alignment, size);
  if (result != 0) {
    fprintf(stderr, "Error allocating memory (%d): %s\n", result,
            strerror(result));
    retval = NULL;
  } else {
    (void)memset(retval, 0x0, size);
  }

  return retval;
}

/// Retreive input/output offsets.
/// \p network_data - Pointer to the NetworkData structure holding metadata for
/// the network. \p in_offset - Output for the input offset (pointer of type
/// size_t*) \p out_offset - Output for the output offset (pointer of type
/// size_t*) \returns X_SUCCESS on success, X_FAILURE on failure (in_offset and
/// out_offset are not valid)
int get_io_offsets(const struct NetworkData *network_data, size_t *in_offset,
                   size_t *out_offset) {
  int retval;
  bool found_in = false;
  bool found_out = false;
  size_t symbol_index;
  size_t input_name_len;
  size_t output_name_len;

  input_name_len = strlen(network_data->input_tensor_name);
  output_name_len = strlen(network_data->output_tensor_name);

  // Look for the input and the output tensors, and grab their offsets. The
  // lookup is done by tensor name.
  for (symbol_index = 0;
       symbol_index < network_data->bundle_config->num_symbols;
       ++symbol_index) {
    if (strncmp(network_data->input_tensor_name,
                network_data->bundle_config->symbols[symbol_index].name,
                input_name_len) == 0) {
      *in_offset = network_data->bundle_config->symbols[symbol_index].offset;
      found_in = true;

      if (found_out) {
        break;
      }
    }
    if (strncmp(network_data->output_tensor_name,
                network_data->bundle_config->symbols[symbol_index].name,
                output_name_len) == 0) {
      *out_offset = network_data->bundle_config->symbols[symbol_index].offset;
      found_out = true;

      if (found_in) {
        break;
      }
    }
  }

  if (!found_in || !found_out) {
    retval = X_FAILURE;
  } else {
    retval = X_SUCCESS;
  }

  return retval;
}
