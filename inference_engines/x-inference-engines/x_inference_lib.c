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
static struct PerfData globalPD;
#endif // ENABLE_PERF_MONITORING

static uint8_t *initConstantWeights(const char *wfname,
                                    const struct BundleConfig *bundleConfig);
static uint8_t *initMutableWeights(const struct BundleConfig *bundleConfig);
static uint8_t *initActivations(const struct BundleConfig *bundleConfig);
static int getIOOffsets(const struct NetworkData *networkData, size_t *inOffset,
                        size_t *outOffset);

static void *mallocAligned(uint64_t alignment, size_t size);

int initRuntimeData(const struct NetworkData *networkData,
                    struct RuntimeData *runtimeData) {
  uint8_t *result = NULL;
  int retval = X_FAILURE;

  (void)memset(runtimeData, 0x0, sizeof(struct RuntimeData));

// If performance monitoring is enabled, make sure to correctly clear/initialize
// the corresponding runtime data.
#ifdef ENABLE_PERF_MONITORING
  runtimeData->doPerfMonitoring = networkData->doPerfMonitoring;
  if (runtimeData->doPerfMonitoring) {
    (void)memset(&(runtimeData->ps), 0x0, sizeof(struct PerfStatistics));
    (void)memset(&globalPD, 0x0, sizeof(struct PerfData));

    // Initialize performance monitor.
    retval = initPerfMonitoring(&globalPD);
    if (retval == -1) {
      perror("ERROR: unable to initialize perf event");
      return X_FAILURE;
    }
  }
#endif // ENABLE_PERF_MONITORING

  result = initConstantWeights(networkData->weightsFileName,
                               networkData->bundleConfig);
  if (result != NULL) {
    runtimeData->constWeights = result;

// If performance monitoring is enabled, report total weights size as part of
// the monitoring.
#ifdef ENABLE_PERF_MONITORING
    if (runtimeData->doPerfMonitoring) {
      globalPD.ps.constWeightsSize =
          networkData->bundleConfig->constWeightVarsMemsize;
    }
#endif // ENABLE_PERF_MONITORING
       // Cascaded initialization of mutable weights->activations->get offsets.
    result = initMutableWeights(networkData->bundleConfig);
    if (result != NULL) {
      runtimeData->mutWeights = result;
      result = initActivations(networkData->bundleConfig);
      if (result != NULL) {
        runtimeData->activations = result;
        runtimeData->inferenceFunc = networkData->inferenceFunction;

        // Where are the input and output tensors located?
        retval = getIOOffsets(networkData, &(runtimeData->inputOffset),
                              &(runtimeData->outputOffset));
      }
    }
  }

  return retval;
}

int initIO(struct InferenceIO *iio, void *inMMap, void *outMMap) {
  int retval = X_FAILURE;

  // If memory-mapped input is passed, we don't need to allocate memory
  // for input. Otherwise, allocate and require cleanup later.
  if (inMMap != NULL) {
    iio->input = inMMap;
    iio->cleanupInput = 0;
  } else {
    iio->input = malloc(iio->batchSize * iio->inLen);
    iio->cleanupInput = 1;
  }

  // If memory-mapped output is passed, we don't need to allocate memory
  // for output. Otherwise, allocate and require cleanup later. This should only
  // be done if we have non-null input.
  if (iio->input != NULL) {
    if (outMMap != NULL) {
      iio->output = outMMap;
      iio->cleanupOutput = 0;
    } else {
      iio->output = malloc(iio->batchSize * iio->outLen);
      iio->cleanupOutput = 1;
    }

    if (iio->output != NULL) {
      retval = X_SUCCESS;
    }
  }

  // This will return X_FAILURE. Success is returned if and only if both input
  // and output are not null (either memory mapped, or successfully allocated)
  return retval;
}

void cleanupRuntimeData(struct RuntimeData *runtimeData) {
  free(runtimeData->activations);
  free(runtimeData->constWeights);
  free(runtimeData->mutWeights);

  runtimeData->activations = NULL;
  runtimeData->constWeights = NULL;
  runtimeData->mutWeights = NULL;

#ifdef ENABLE_PERF_MONITORING
  if (runtimeData->doPerfMonitoring) {
    (void)stopPerfMonitoring(&globalPD);
  }
#endif // ENABLE_PERF_MONITORING
}

void cleanupIO(struct InferenceIO *iio) {
  if (iio->cleanupInput) {
    free(iio->input);
  }
  if (iio->cleanupOutput) {
    free(iio->output);
  }

  iio->input = NULL;
  iio->output = NULL;
}

void runInference(const struct InferenceIO *iio,
                  struct RuntimeData *runtimeData) {
  size_t batchCounter;
  size_t currentInputOffset;
  size_t currentOutputOffset;

  if (iio->batchSize == 0) {
    return;
  }

  currentInputOffset = 0;
  currentOutputOffset = 0;

#ifdef ENABLE_PERF_MONITORING
  if (runtimeData->doPerfMonitoring) {
    globalPD.ps.numCases = iio->batchSize;
  }
#endif // ENABLE_PERF_MONITORING

  for (batchCounter = 0; batchCounter < iio->batchSize; ++batchCounter) {
    // Store the next input into the correct mutable weights location.
    (void)memcpy(runtimeData->mutWeights + runtimeData->inputOffset,
                 iio->input + currentInputOffset, iio->inLen);

#ifdef ENABLE_PERF_MONITORING
    if (runtimeData->doPerfMonitoring) {
      (void)resumePerfMonitoring(&globalPD);
    }
#endif // ENABLE_PERF_MONITORING
       // Call our inference function on the input data. Pass in locations of
       // constant weights, mutable weights (inputs/outputs), and activations.
    (runtimeData->inferenceFunc)(runtimeData->constWeights,
                                 runtimeData->mutWeights,
                                 runtimeData->activations);
#ifdef ENABLE_PERF_MONITORING
    if (runtimeData->doPerfMonitoring) {
      (void)pausePerfMonitoring(&globalPD);
      (void)readPerfStatistics(&globalPD);
      (void)memcpy(&(runtimeData->ps), &(globalPD.ps),
                   sizeof(struct PerfStatistics));
    }
#endif // ENABLE_PERF_MONITORING

    // Store the current output into the correct output location.
    (void)memcpy(iio->output + currentOutputOffset,
                 runtimeData->mutWeights + runtimeData->outputOffset,
                 iio->outLen);

    // Advance the input/output pointers to the next position of the next input
    // in the batch, and the corresponding output.
    currentInputOffset += iio->inLen;
    currentOutputOffset += iio->outLen;
  }
}

/// Initialize constant weights with the data from the weights file.
/// \p wfname - Constant weights file name.
/// \p bundleConfig - The bundle config structure
/// \returns Pointer to the constant weights on success, NULL on failure.
uint8_t *initConstantWeights(const char *wfname,
                             const struct BundleConfig *bundleConfig) {
  size_t size = 0;
  int fd = 0;
  off_t fileOffset = 0;
  uint8_t *retval = NULL;
  uint8_t *buffer = NULL;
  int bytesRead = 0;
  size_t bytesTotal = 0;

  fd = open(wfname, O_RDONLY);
  if (fd == -1) {
    perror("Error processing weights file");
    return NULL;
  }

  fileOffset = lseek(fd, 0, SEEK_END);
  if (fileOffset == -1) {
    perror("Error processing weights file");
  } else {
    // Make sure the weights file is of expected size!
    size = (size_t)(fileOffset);
    if (size != bundleConfig->constWeightVarsMemsize) {
      fprintf(stderr,
              "Unexpected file size (%zd) does not match expected (%llu)\n",
              size, bundleConfig->constWeightVarsMemsize);

      (void)close(fd);
      return NULL;
    }

    fileOffset = lseek(fd, 0, SEEK_SET);
    if (fileOffset == -1) {
      perror("Error processing weights file");

      (void)close(fd);
      return NULL;
    }

    retval = mallocAligned(bundleConfig->alignment, size);
    buffer = retval;
    // Read in the weights.
    if (retval != NULL) {
      while (bytesTotal < size) {
        bytesRead = read(fd, buffer, size - bytesTotal);
        bytesTotal += bytesRead;

        if (bytesRead <= 0) {
          if (bytesRead == -1) {
            perror("Error reading weights file");
          } else if (bytesRead == 0) {
            fprintf(stderr,
                    "Error reading weights file: EOF reached too early\n");
          }

          free(retval);
          retval = NULL;
          break;
        }

        buffer += bytesRead;
      }
    } else
      perror("Error allocating memory for weights");
  }

  (void)close(fd);
  return retval;
}

/// Initialize mutable weights -- simply allocates the memory.
/// \returns Valid pointer on success, NULL on failure.
uint8_t *initMutableWeights(const struct BundleConfig *bundleConfig) {
  return mallocAligned(bundleConfig->alignment,
                       bundleConfig->mutWeightVarsMemsize);
}

/// Initialize activations -- simply allocates the memory.
/// \returns Valid pointer on success, NULL on failure.
uint8_t *initActivations(const struct BundleConfig *bundleConfig) {
  return mallocAligned(bundleConfig->alignment,
                       bundleConfig->activationMemsize);
}

/// Performs aligned memory allocation.
/// \p alignment - alignment requirement
/// \p size - size of memory block to allocate
/// \returns Valid pointer on success, NULL on failure.
void *mallocAligned(uint64_t alignment, size_t size) {
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

/// Retrieve input/output offsets.
/// \p networkData - Pointer to the NetworkData structure holding metadata for
/// the network. \p inOffset - Output for the input offset (pointer of type
/// size_t*) \p outOffset - Output for the output offset (pointer of type
/// size_t*) \returns X_SUCCESS on success, X_FAILURE on failure (inOffset and
/// outOffset are not valid)
int getIOOffsets(const struct NetworkData *networkData, size_t *inOffset,
                 size_t *outOffset) {
  int retval;
  bool foundIn = false;
  bool foundOut = false;
  size_t symbolIndex;
  size_t inputNameLen;
  size_t outputNameLen;

  inputNameLen = strlen(networkData->inputTensorName);
  outputNameLen = strlen(networkData->outputTensorName);

  // Look for the input and the output tensors, and grab their offsets. The
  // lookup is done by tensor name.
  for (symbolIndex = 0; symbolIndex < networkData->bundleConfig->numSymbols;
       ++symbolIndex) {
    if (strncmp(networkData->inputTensorName,
                networkData->bundleConfig->symbols[symbolIndex].name,
                inputNameLen) == 0) {
      *inOffset = networkData->bundleConfig->symbols[symbolIndex].offset;
      foundIn = true;

      if (foundOut) {
        break;
      }
    }
    if (strncmp(networkData->outputTensorName,
                networkData->bundleConfig->symbols[symbolIndex].name,
                outputNameLen) == 0) {
      *outOffset = networkData->bundleConfig->symbols[symbolIndex].offset;
      foundOut = true;

      if (foundIn) {
        break;
      }
    }
  }

  if (!foundIn || !foundOut) {
    retval = X_FAILURE;
  } else {
    retval = X_SUCCESS;
  }

  return retval;
}
