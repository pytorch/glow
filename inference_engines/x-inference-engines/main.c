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

#include <argp.h>
#include <assert.h>
#include <dlfcn.h>
#include <fcntl.h>
#include <stdio.h>
#include <string.h>
#include <unistd.h>

#include "x_inference_lib.h"

#ifndef X_USE_DYNAMIC

#ifndef X_MODEL_NAME
#error "X_MODEL_NAME must be defined when using static bundles!"
#endif // X_MODEL_NAME

#define GLUE(m, c) m##_##c
#define BUILD_SYMBOL_NAME(m, c) GLUE(m, c)

extern struct BundleConfig BUILD_SYMBOL_NAME(X_MODEL_NAME, config);
extern void X_MODEL_NAME(uint8_t *, uint8_t *, uint8_t *);

#endif // X_USE_DYNAMIC

const char *argpProgramVersion = "x-infer v0.01";
const char *argpProgramBugAddress =
    "Github Pytorch Glow repository at https://github.com/pytorch/glow";
const char doc[] =
    "\n                    Generic Inference Engine                         \n"
    "-----------------------------------------------------------------------\n"
#ifdef X_USE_DYNAMIC
    "Dynamic bundle loading: SUPPORTED\n"
#else
    "Dynamic bundle loading: NOT SUPPORTED (bundle has been statically "
    "linked)\n"
#endif // X_USE_DYNAMIC
#ifdef ENABLE_PERF_MONITORING
    "Performance monitoring: SUPPORTED\n"
#else
    "Performance monitoring: NOT SUPPORTED\n"
#endif // ENABLE_PERF_MONITORING
    "\nx-infer runs inference against the provided Glow bundle. Weights file "
    "must be "
    "specified as the first argument. The input file must be specified with "
    "[--infile] (binary file). Input tensor type [-intype], output tensor type "
    "[-outtype], "
    "input length [--inlen], output length [--outlen], input tensor name "
    "[-inname], "
    "and output tensor name [--outname] must be specified. "
    "\n\n"
    "When built with dynamic bundle loading support, bundle must be specified "
    "as the "
    "first positional argument, and the weights file as the second. When built "
    "with "
    "a bundle statically linked in, dynamic loading is not supported "
#ifdef ENABLE_PERF_MONITORING
    "\n\nAdditionally, performance monitoring is available with [--perf] "
    "option. In this case, "
    "[--perflog] can be used to specify the performance log filename. "
    "Otherwise, performance log "
    "is written to stdout "
#endif // ENABLE_PERF_MONITORING

    "\n\nShort and long form options are: ";
const char argsDoc[] =
#ifdef X_USE_DYNAMIC
    "[BUNDLE FILENAME] "
#endif // X_USE_DYNAMIC
    "[WEIGHTS FILENAME]";

const struct argp_option options[] = {
    {"output", 'o', "FILE", 0,
     "Output to binary FILE instead of standard output"},
    {"infile", 'i', "FILE", 0, "Input from FILE"},
    {"intype", 't', "TYPE", 0, "Input TYPE (one of F32, F16, I16, I8)"},
    {"outtype", 'T', "TYPE", 0, "Output TYPE (one of F32, F16, I16, I8)"},
    {"inlen", 'l', "LEN", 0,
     "Input tensor length (e.g. if the tensor is of shape 2x3x4, its "
     "length is 2 * 3 * 4 = 24)"},
    {"outlen", 'L', "LEN", 0,
     "Output tensor length (e.g. if the tensor is of shape 2x3x4, its "
     "length is 2 * 3 * 4 = 24)"},
    {"inname", 'n', "NAME", 0, "Input tensor name NAME"},
    {"outname", 'N', "NAME", 0, "Output tensor name NAME"},

#ifdef ENABLE_PERF_MONITORING
    {"perf", 'p', 0, 0, "Whether to output performance logs"},
    {"perflog", 'P', "NAME", 0,
     "Performance log output filename (stdout if omitted)"},
#endif // ENABLE_PERF_MONITORING

#ifdef X_USE_DYNAMIC
    {"model", 'm', "NAME", 0, "Model name (maximum 128 chars)"},
#endif // X_USE_DYNAMIC

    {0}};

struct Arguments {
#ifdef X_USE_DYNAMIC
  char *bundleFile;
  char bundleConfigName[135];
  char *modelName;
#endif // X_USE_DYNAMIC
#ifdef ENABLE_PERF_MONITORING
  int perfMonitor;
  char *perfLogFile;
#endif // ENABLE_PERF_MONITORING
  char *weightsFile;
  char *outFile;
  char *inFile;
  char *inType;
  char *outType;
  char *inLen;
  char *outLen;
  char *inName;
  char *outName;
  size_t inTensorSize;
  size_t outTensorSize;
  size_t batch;
};

static error_t parseOpt(int key, char *arg, struct argp_state *state);
static void initArguments(struct Arguments *arguments);
static int computeArguments(struct Arguments *arguments);
static void cleanupArguments(struct Arguments *arguments);
static int retreiveAndLoadInput(struct InferenceIO *inferenceIO,
                                const struct Arguments *arguments);
static int retreiveAndStoreOutput(struct InferenceIO *inferenceIO,
                                  const struct Arguments *arguments);

#ifdef ENABLE_PERF_MONITORING
static void reportPerformance(const struct PerfStatistics *ps,
                              const char *filename);
#endif // ENABLE_PERF_MONITORING

static struct argp argp = {options, parseOpt, argsDoc, doc};

int main(int argc, char **argv) {
  struct Arguments arguments;
  struct NetworkData networkData;
  struct RuntimeData runtimeData;
  struct InferenceIO inferenceIO;
  int retval;

// Using dynamic loading? If so, will load with dload.
#ifdef X_USE_DYNAMIC
  struct BundleConfig *xConfig;
  InferenceFunctionPtr_t xInfer;
  void *dlHandle;
#endif // X_USE_DYNAMIC

  initArguments(&arguments);
  retval = argp_parse(&argp, argc, argv, 0, 0, &arguments);
  if (retval != X_SUCCESS) {
    fprintf(stderr, "ERROR: Invalid arguments. Use --help for help.");
    cleanupArguments(&arguments);
    exit(retval);
  }

  retval = computeArguments(&arguments);
  if (retval != X_SUCCESS) {
    cleanupArguments(&arguments);
    exit(retval);
  }

#ifdef X_USE_DYNAMIC
  // Load the bundle...
  dlHandle = dlopen(arguments.bundleFile, RTLD_NOW);
  if (dlHandle == NULL) {
    fprintf(stderr, "ERROR: Cannot load bundle %s: %s\n", arguments.bundleFile,
            dlerror());
    cleanupArguments(&arguments);
    exit(X_FAILURE);
  }

  xConfig = dlsym(dlHandle, arguments.bundleConfigName);
  if (dlerror() != NULL) {
    fprintf(stderr,
            "ERROR: Cannot load bundle config structure %s from %s: %s.\n",
            arguments.bundleConfigName, arguments.bundleFile, dlerror());
    cleanupArguments(&arguments);
    exit(X_FAILURE);
  }

  xInfer = dlsym(dlHandle, arguments.modelName);
  if (dlerror() != NULL) {
    fprintf(stderr, "ERROR: Cannot load model %s from %s: %s.\n",
            arguments.modelName, arguments.bundleFile, dlerror());
    cleanupArguments(&arguments);
    exit(X_FAILURE);
  }
#endif // X_USE_DYFNAMIC

  // Load the network data...
#ifdef X_USE_DYNAMIC
  networkData.bundleConfig = xConfig;
  networkData.inferenceFunction = xInfer;
#else
  // Not loading dynamically - so build the symbol names from the #define's.
  networkData.bundleConfig = &BUILD_SYMBOL_NAME(X_MODEL_NAME, config);
  networkData.inferenceFunction = X_MODEL_NAME;
#endif // X_USE_DYNAMIC

  networkData.inputTensorName = arguments.inName;
  networkData.outputTensorName = arguments.outName;
  networkData.weightsFileName = arguments.weightsFile;
#ifdef ENABLE_PERF_MONITORING
  networkData.doPerfMonitoring = arguments.perfMonitor;
#endif // ENABLE_PERF_MONITORING

  // Initialize the runtime data...
  retval = initRuntimeData(&networkData, &runtimeData);
  if (retval != X_SUCCESS) {
    fprintf(stderr, "ERROR: Could not initialize data. Exiting!\n");
    cleanupRuntimeData(&runtimeData);
    cleanupArguments(&arguments);
    exit(retval);
  }

  // Initialize the IO struct...
  inferenceIO.inLen = arguments.inTensorSize;
  inferenceIO.outLen = arguments.outTensorSize;
  inferenceIO.batchSize = arguments.batch;

  retval = initIO(&inferenceIO, NULL, NULL);
  if (retval != X_SUCCESS) {
    fprintf(stderr, "ERROR: Could not initialize IO. Exiting!\n");
    cleanupRuntimeData(&runtimeData);
    cleanupIO(&inferenceIO);
    cleanupArguments(&arguments);
    exit(retval);
  }

  retval = retreiveAndLoadInput(&inferenceIO, &arguments);
  if (retval != X_SUCCESS) {
    cleanupRuntimeData(&runtimeData);
    cleanupIO(&inferenceIO);
    cleanupArguments(&arguments);
    exit(retval);
  }

  runInference(&inferenceIO, &runtimeData);

  retval = retreiveAndStoreOutput(&inferenceIO, &arguments);
  if (retval != X_SUCCESS) {
    cleanupRuntimeData(&runtimeData);
    cleanupIO(&inferenceIO);
    cleanupArguments(&arguments);
    exit(retval);
  }

#ifdef ENABLE_PERF_MONITORING
  if (runtimeData.doPerfMonitoring) {
    reportPerformance(&(runtimeData.ps), arguments.perfLogFile);
  }
#endif // ENABLE_PERF_MONITORING

  // Finally, don't forget to clean up...
  cleanupRuntimeData(&runtimeData);
  cleanupIO(&inferenceIO);
  cleanupArguments(&arguments);

  exit(0);
}

/// Parses command line options given the short form \p key, its argument \p
/// arg, and the arg_state \p state
error_t parseOpt(int key, char *arg, struct argp_state *state) {
  struct Arguments *arguments = state->input;

  switch (key) {
  case 'o':
    arguments->outFile = arg;
    break;
  case 'i':
    arguments->inFile = arg;
    break;
  case 't':
    arguments->inType = arg;
    break;
  case 'T':
    arguments->outType = arg;
    break;
  case 'l':
    arguments->inLen = arg;
    break;
  case 'L':
    arguments->outLen = arg;
    break;
  case 'n':
    arguments->inName = arg;
    break;
  case 'N':
    arguments->outName = arg;
    break;
#ifdef ENABLE_PERF_MONITORING
  case 'p':
    arguments->perfMonitor = 1;
    break;
  case 'P':
    arguments->perfLogFile = arg;
    break;
#endif // ENABLE_PERF_MONITORING
#ifdef X_USE_DYNAMIC
  case 'm':
    arguments->modelName = arg;
    break;
#endif // X_USE_DYNAMIC
  case ARGP_KEY_ARG:
#ifdef X_USE_DYNAMIC
    if (state->arg_num >= 3) {
      argp_usage(state);
    }
    if (state->arg_num == 1) {
      arguments->weightsFile = arg;
    } else {
      arguments->bundleFile = arg;
    }
#else
    if (state->arg_num >= 2) {
      argp_usage(state);
    }
    arguments->weightsFile = arg;
#endif // X_USE_DYNAMIC
    break;
  case ARGP_KEY_END:
#ifdef X_USE_DYNAMIC
    if (state->arg_num < 2) {
#else
    if (state->arg_num < 1) {
#endif // X_USE_DYNAMIC
      argp_usage(state);
    }
    break;
  default:
    return ARGP_ERR_UNKNOWN;
  }

  return X_SUCCESS;
}

/// Clear the memory for argument values \p arguments.
void initArguments(struct Arguments *arguments) {
  memset(arguments, 0x0, sizeof(struct Arguments));
}

/// Initialize argument values \p arguments based on the passed arguments.
int computeArguments(struct Arguments *arguments) {
  int fd;
  off_t fileOffset;
  size_t inputSize;
  size_t inTensorTypeSize;
  size_t outTensorTypeSize;
  long inTensorLen;
  long outTensorLen;

#ifdef X_USE_DYNAMIC
  if (arguments->modelName == NULL) {
    fprintf(stderr, "ERROR: -m option must be specified.\n");
    return X_FAILURE;
  }
  sprintf(arguments->bundleConfigName, "%s_%s", arguments->modelName, "config");
#endif // X_USE_DYNAMIC

  if (arguments->inType == NULL || arguments->outType == NULL) {
    fprintf(stderr, "ERROR: -t and -T must be specified.\n");
    return X_FAILURE;
  }

  if (strncmp(arguments->inType, "F32", 4) == 0) {
    inTensorTypeSize = 4;
  } else if (strncmp(arguments->inType, "F16", 4) == 0 ||
             strncmp(arguments->inType, "I16", 4) == 0) {
    inTensorTypeSize = 2;
  } else if (strncmp(arguments->inType, "I8", 3) == 0) {
    inTensorTypeSize = 1;
  } else {
    fprintf(stderr, "ERROR: Invalid input tensor type %s\n", arguments->inType);
    return X_FAILURE;
  }

  if (strncmp(arguments->outType, "F32", 4) == 0) {
    outTensorTypeSize = 4;
  } else if (strncmp(arguments->outType, "F16", 4) == 0 ||
             strncmp(arguments->outType, "I16", 4) == 0) {
    outTensorTypeSize = 2;
  } else if (strncmp(arguments->outType, "I8", 3) == 0) {
    outTensorTypeSize = 1;
  } else {
    fprintf(stderr, "ERROR: Invalid output tensor type %s\n",
            arguments->outType);
    return X_FAILURE;
  }

  if (arguments->inLen == NULL || arguments->outLen == NULL) {
    fprintf(stderr, "ERROR: -l and -L options must be specified.\n");
    return X_FAILURE;
  }

  inTensorLen = atol(arguments->inLen);
  outTensorLen = atol(arguments->outLen);
  if (inTensorLen <= 0) {
    fprintf(stderr, "ERROR: Invalid -l value: %s.\n", arguments->inLen);
    return X_FAILURE;
  }
  if (outTensorLen <= 0) {
    fprintf(stderr, "ERROR: Invalid -L value: %s.\n", arguments->outLen);
    return X_FAILURE;
  }

  arguments->inTensorSize = (size_t)(inTensorLen)*inTensorTypeSize;
  arguments->outTensorSize = (size_t)(outTensorLen)*outTensorTypeSize;

  if (arguments->inName == NULL || arguments->outName == NULL) {
    fprintf(stderr, "ERROR: both -n and -N options must be specified.\n");
    return X_FAILURE;
  }

  if (arguments->inFile == NULL) {
    fprintf(stderr, "ERROR: -i option must be specified.\n");
    return X_FAILURE;
  }

  if (arguments->outFile == NULL) {
    fprintf(stderr, "ERROR: -o option must be specified.\n");
    return X_FAILURE;
  }

  fd = open(arguments->inFile, O_RDONLY);
  if (fd == -1) {
    perror("ERROR: Could not process input file");
    return X_FAILURE;
  }

  fileOffset = lseek(fd, 0, SEEK_END);
  if (fileOffset == -1) {
    perror("ERRPR: Could not process input file");
    (void)close(fd);
    return X_FAILURE;
  }
  inputSize = (size_t)(fileOffset);

  if (inputSize % arguments->inTensorSize != 0) {
    fprintf(stderr,
            "ERROR: Input file size (%zu bytes) is not a multiple of input "
            "tensor size (%zu bytes).\n",
            inputSize, arguments->inTensorSize);
    return X_FAILURE;
  }

  arguments->batch = inputSize / arguments->inTensorSize;

  return X_SUCCESS;
}

/// Currently the same as initArguments().
void cleanupArguments(struct Arguments *arguments) { initArguments(arguments); }

/// Read input from the input file held in \p arguments, and load it into \p
/// inferenceIO for inference. \returns X_SUCCESS on success, X_FAILURE on
/// failure.
int retreiveAndLoadInput(struct InferenceIO *inferenceIO,
                         const struct Arguments *arguments) {
  const size_t size = arguments->batch * arguments->inTensorSize;
  size_t bytesTotal;
  int bytesRead;
  int fd;
  int retval = X_SUCCESS;

  fd = open(arguments->inFile, O_RDONLY);
  if (fd == -1) {
    perror("ERROR: Could not process input file");
    return X_FAILURE;
  }

  bytesTotal = 0;
  while (bytesTotal < size) {
    bytesRead = read(fd, inferenceIO->input, size - bytesTotal);
    bytesTotal += bytesRead;

    if (bytesRead <= 0) {
      if (bytesRead == -1) {
        perror("ERROR: Could not read input file");
      } else if (bytesRead == 0) {
        fprintf(stderr,
                "ERROR: Could not read input file - EOF reached too early.\n");
      }

      retval = X_FAILURE;
      break;
    }
  }
  (void)close(fd);

  return retval;
}

/// Retreives output held in \p inferenceIO and writes it to the output file
/// whose name is stored in \p arguments. \returns X_SUCCESS on success,
/// X_FAILURE on failure.
int retreiveAndStoreOutput(struct InferenceIO *inferenceIO,
                           const struct Arguments *arguments) {
  const size_t size = arguments->batch * arguments->outTensorSize;
  size_t bytesTotal;
  int bytesWritten;
  uint8_t *buffer;
  int fd;
  int retval = X_SUCCESS;

  fd = open(arguments->outFile, O_CREAT | O_WRONLY, S_IRUSR | S_IWUSR);
  if (fd == -1) {
    perror("ERROR: Could not open output file");
    return X_FAILURE;
  }

  bytesTotal = 0;
  buffer = inferenceIO->output;
  while (bytesTotal < size) {
    bytesWritten = write(fd, buffer, size - bytesTotal);
    bytesTotal += bytesWritten;

    if (bytesWritten <= 0) {
      if (bytesWritten == -1) {
        perror("ERROR: Could not write to output file");
        retval = X_FAILURE;
        break;
      }
      // Technically, the device could be busy. We'll then retry indefinitely.
      // Is this the best option? This is really architecture dependent. In most
      // sane scenarios, this should never happen with regular files.
      else if (bytesWritten == 0) {
        fprintf(stderr,
                "WARNING: Wrote 0 bytes (is device busy? will retry).\n");
      }
    }

    buffer += bytesWritten;
  }
  (void)close(fd);

  return retval;
}

#ifdef ENABLE_PERF_MONITORING
void reportPerformance(const struct PerfStatistics *ps, const char *filename) {
  int fd;
  const size_t outputBufferSize = 512;
  char buffer[outputBufferSize] = {0};
  int bytesWritten;
  size_t bytesTotal;

  snprintf(buffer, outputBufferSize,
           "\nConstant weights size       : %zd bytes\n"
           "Number of cases             : %zd\n"
           "Number of CPU cycles (x1-e6): %f\n\n",
           ps->constWeightsSize, ps->numCases, ps->numCPUCycles / 1.0e6);

  if (filename != NULL) {
    fd = open(filename, O_CREAT | O_WRONLY, S_IRUSR | S_IWUSR);
    if (fd == -1) {
      perror("ERROR: Could not open perf log file for writing");
      return;
    }
  } else {
    fd = STDOUT_FILENO;
  }

  bytesTotal = 0;
  while (bytesTotal < outputBufferSize) {
    bytesWritten = write(fd, buffer, outputBufferSize - bytesTotal);
    bytesTotal += bytesWritten;

    if (bytesWritten <= 0) {
      if (bytesWritten == -1) {
        perror("ERROR: Could not write to perf log file");
        break;
      } else if (bytesWritten == 0) {
        fprintf(stderr,
                "WARNING: Wrote 0 bytes (is device busy? will retry).\n");
      }
    }
  }

  if (fd != STDOUT_FILENO) {
    (void)close(fd);
  }
}
#endif // ENABLE_PERF_MONITORING
