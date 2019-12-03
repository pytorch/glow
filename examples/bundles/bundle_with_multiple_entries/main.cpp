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
#include <assert.h>
#include <inttypes.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <string>
#include <vector>

#include "testBundle.h"

//===----------------------------------------------------------------------===//
//                 Wrapper code for executing a bundle
//===----------------------------------------------------------------------===//
/// Find in the bundle's symbol table a weight variable whose name starts with
/// \p name.
const SymbolTableEntry *getWeightVar(const BundleConfig &config,
                                     const char *name) {
  for (unsigned i = 0, e = config.numSymbols; i < e; ++i) {
    if (!strncmp(config.symbolTable[i].name, name, strlen(name))) {
      return &config.symbolTable[i];
    }
  }
  return nullptr;
}

/// Find in the bundle's symbol table a mutable weight variable whose name
/// starts with \p name.
const SymbolTableEntry &getMutableWeightVar(const BundleConfig &config,
                                            const char *name) {
  const SymbolTableEntry *mutableWeightVar = getWeightVar(config, name);
  if (!mutableWeightVar) {
    printf("Expected to find variable '%s'\n", name);
  }
  assert(mutableWeightVar && "Expected to find a mutable weight variable");
  assert(mutableWeightVar->kind != 0 &&
         "Weight variable is expected to be mutable");
  return *mutableWeightVar;
}

/// Allocate an aligned block of memory.
void *alignedAlloc(const BundleConfig &config, size_t size) {
  void *ptr;
  // Properly align the memory region.
  int res = posix_memalign(&ptr, config.alignment, size);
  assert(res == 0 && "posix_memalign failed");
  assert((size_t)ptr % config.alignment == 0 && "Wrong alignment");
  memset(ptr, 0, size);
  (void)res;
  return ptr;
}

/// Initialize the constant weights memory block by loading the weights from the
/// weights file.
static uint8_t *initConstantWeights(const char *weightsFileName,
                                    const BundleConfig &config) {
  // Load weights.
  FILE *weightsFile = fopen(weightsFileName, "rb");
  if (!weightsFile) {
    fprintf(stderr, "Could not open the weights file: %s\n", weightsFileName);
    exit(1);
  }
  fseek(weightsFile, 0, SEEK_END);
  size_t fileSize = ftell(weightsFile);
  fseek(weightsFile, 0, SEEK_SET);
  uint8_t *baseConstantWeightVarsAddr =
      static_cast<uint8_t *>(alignedAlloc(config, fileSize));
  printf("Allocated weights of size: %lu\n", fileSize);
  printf("Expected weights of size: %" PRIu64 "\n",
         config.constantWeightVarsMemSize);
  assert(fileSize == config.constantWeightVarsMemSize &&
         "Wrong weights file size");
  int result = fread(baseConstantWeightVarsAddr, fileSize, 1, weightsFile);
  if (result != 1) {
    perror("Could not read the weights file");
  } else {
    printf("Loaded weights of size: %lu from the file %s\n", fileSize,
           weightsFileName);
  }
  fclose(weightsFile);
  float *dataPtr = (float *)baseConstantWeightVarsAddr;
  printf("Constants:");
  for (size_t idx = 0, e = fileSize / sizeof(float); idx < e; ++idx) {
    if (idx % 8 == 0) {
      printf("\n");
      printf("offset %4ld: ", idx * sizeof(float));
    }
    printf(" %f", dataPtr[idx]);
  }
  printf("\n");
  return baseConstantWeightVarsAddr;
}

static uint8_t *allocateMutableWeightVars(const BundleConfig &config) {
  auto *weights = static_cast<uint8_t *>(
      alignedAlloc(config, config.mutableWeightVarsMemSize));
  printf("Allocated mutable weight variables of size: %" PRIu64 "\n",
         config.mutableWeightVarsMemSize);
  return weights;
}

/// Dump the result of the inference by looking at the results vector and
/// finding the index of the max element.
static void dumpInferenceResult(const BundleConfig &config,
                                uint8_t *mutableWeightVars, const char *name) {
  const SymbolTableEntry &outputWeight = getMutableWeightVar(config, name);
  float *outputWeightPtr =
      reinterpret_cast<float *>(mutableWeightVars + outputWeight.offset);
  printf("Output weight %s:", name);
  for (size_t idx = 0, e = outputWeight.size; idx < e; ++idx) {
    printf(" %f", outputWeightPtr[idx]);
  }
  printf("\n");
}

static uint8_t *initMutableWeightVars(const BundleConfig &config,
                                      const char *name) {
  uint8_t *mutableWeightVarsAddr = allocateMutableWeightVars(config);
  const SymbolTableEntry &inputVar = getMutableWeightVar(config, name);

  float *inputVar1Ptr =
      reinterpret_cast<float *>(mutableWeightVarsAddr + inputVar.offset);
  for (size_t idx = 0, e = inputVar.size; idx < e; ++idx) {
    inputVar1Ptr[idx] = 1.0f;
  }
  return mutableWeightVarsAddr;
}

static uint8_t *initActivations(const BundleConfig &config) {
  return static_cast<uint8_t *>(
      alignedAlloc(config, config.activationsMemSize));
}

/// Invoke \p bundleEntry with the input \p inputName and output \p outputName.
void testBundleEntry(const char *outputName, const char *inputName,
                     void (*bundleEntry)(uint8_t *constantWeight,
                                         uint8_t *mutableWeight,
                                         uint8_t *activations)) {
  // Allocate and initialize constant and mutable weights.
  uint8_t *constantWeightVarsAddr =
      initConstantWeights("testBundle.weights.bin", testBundle_config);
  uint8_t *mutableWeightVarsAddr =
      initMutableWeightVars(testBundle_config, inputName);
  uint8_t *activationsAddr = initActivations(testBundle_config);

  // Perform the computation.
  bundleEntry(constantWeightVarsAddr, mutableWeightVarsAddr, activationsAddr);

  // Report the results.
  dumpInferenceResult(testBundle_config, mutableWeightVarsAddr, outputName);

  // Free all resources.
  free(activationsAddr);
  free(constantWeightVarsAddr);
  free(mutableWeightVarsAddr);
}

int main(int argc, char **argv) {
  // Invoke all entry points of the bundle.
  testBundleEntry("output1", "input1", testMainEntry1);
  testBundleEntry("output2", "input2", testMainEntry2);
}
