#ifndef X_INFERENCE_LIB_H
#define X_INFERENCE_LIB_H

#include <stdlib.h>
#include <stdint.h>

#define X_SUCCESS (0)
#define X_FAILURE (-1)

typedef void (*InferenceFunctionPtr_t)(uint8_t *, uint8_t *, uint8_t*);

struct SymbolTableEntry
{
    const char *name;
    uint64_t offset;
    uint64_t size;
    char kind;
};

struct BundleConfig
{
    uint64_t const_weight_vars_memsize;
    uint64_t mut_weight_vars_memsize;
    uint64_t activation_memsize;
    uint64_t alignment;
    uint64_t num_symbols;
    const struct SymbolTableEntry *symbols;
};

struct RuntimeData
{
    uint8_t *const_weights;
    uint8_t *mut_weights;
    uint8_t *activations;
    InferenceFunctionPtr_t inference_func;
    size_t input_offset;
    size_t output_offset;
};

struct InferenceIO
{
    void *input;
    void *output;
    int cleanup_input;
    int cleanup_output;
    size_t in_len;
    size_t out_len;
    size_t batch_size;
};

struct NetworkData
{
    struct BundleConfig *bundle_config;
    char *input_tensor_name;
    char *output_tensor_name;
    char *weights_file_name;
    InferenceFunctionPtr_t inference_function;
};

int init_runtime_data(const struct NetworkData *network_data, struct RuntimeData *runtime_data);
int init_io(struct InferenceIO *iio, void *in_mmap, void *out_mmap);
void cleanup_runtime_data(struct RuntimeData *runtime_data);
void cleanup_io(struct InferenceIO *io);
void run_inference(const struct InferenceIO *iio, const struct RuntimeData *runtime_data);

#endif