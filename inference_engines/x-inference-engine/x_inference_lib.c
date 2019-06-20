/* Needed to expose (and silence warnings about implicit declaration of) posix_memalign(). */
#define _POSIX_C_SOURCE (200112L)

#include <stdio.h>
#include <memory.h>
#include <string.h>
#include <unistd.h>
#include <errno.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <stdbool.h>

#ifdef ENABLE_PERF_MONITORING
#include <x_perf_monitor.h>
#endif

#include "x_inference_lib.h"

#ifdef ENABLE_PERF_MONITORING
static struct PerfData global_pd;
#endif

static uint8_t *init_constant_weights(const char *wfname, const struct BundleConfig *bundle_config);
static uint8_t *init_mutable_weights(const struct BundleConfig *bundle_config);
static uint8_t *init_activations(const struct BundleConfig *bundle_config);
static int get_io_offsets(const struct NetworkData *network_data, size_t *in_offset, size_t *out_offset);

static void *malloc_aligned(uint64_t alignment, size_t size);

int 
init_runtime_data(const struct NetworkData *network_data, struct RuntimeData *runtime_data)
{
    uint8_t *result = NULL;
    int retval = X_FAILURE;

    (void) memset(runtime_data, 0x0, sizeof(struct RuntimeData));

#ifdef ENABLE_PERF_MONITORING
    (void) memset(&(runtime_data->ps), 0x0, sizeof(struct PerfStatistics));
    (void) memset(&global_pd, 0x0, sizeof(struct PerfData));

    retval = init_perf_monitoring(&global_pd);
    if (retval == -1) {
        perror("ERROR: unable to initialize perf event");
        return X_FAILURE;
    }
#endif

    result = init_constant_weights(network_data->weights_file_name, network_data->bundle_config);
    if (result != NULL) {
        runtime_data->const_weights = result;

#ifdef ENABLE_PERF_MONITORING
        global_pd.ps.const_weights_size = network_data->bundle_config->const_weight_vars_memsize;
#endif

        result = init_mutable_weights(network_data->bundle_config);
        if (result != NULL) {
            runtime_data->mut_weights = result;
            result = init_activations(network_data->bundle_config);
            if (result != NULL) {
                runtime_data->activations = result;
                runtime_data->inference_func = network_data->inference_function;

                retval = get_io_offsets(
                    network_data, &(runtime_data->input_offset),
                    &(runtime_data->output_offset));
            }
        }
    }

    return retval;
}

int 
init_io(struct InferenceIO *iio, void *in_mmap, void *out_mmap)
{
    int retval = X_FAILURE;

    if (in_mmap != NULL) {
        iio->input = in_mmap;
        iio->cleanup_input = 0;
    } else {
        iio->input = malloc(iio->batch_size * iio->in_len);
        iio->cleanup_output = 1;
    }

    if (iio->input != NULL) {
        if (out_mmap != NULL) {
            iio->output = out_mmap;
            iio->cleanup_output = 0;
         } else {
            iio->output = malloc(iio->batch_size * iio->out_len);
            iio->cleanup_output = 1;
         }

        if (iio->output != NULL)
            retval = X_SUCCESS;
    }

    return retval;
}

void 
cleanup_runtime_data(struct RuntimeData *runtime_data)
{
    free(runtime_data->activations);
    free(runtime_data->const_weights);
    free(runtime_data->mut_weights);

    runtime_data->activations = NULL;
    runtime_data->const_weights = NULL;
    runtime_data->mut_weights = NULL;


#ifdef ENABLE_PERF_MONITORING
    (void) stop_perf_monitoring(&global_pd);
#endif
}

void 
cleanup_io(struct InferenceIO *iio)
{
    if (iio->cleanup_input)
        free(iio->input);
    if (iio->cleanup_output)
        free(iio->output);

    iio->input = NULL;
    iio->output = NULL;
}

void 
run_inference(const struct InferenceIO *iio, struct RuntimeData *runtime_data)
{
    size_t batch_counter;
    size_t current_input_offset;
    size_t current_output_offset;

    if (iio->batch_size == 0)
        return;

    current_input_offset = 0;
    current_output_offset = 0;

#ifdef ENABLE_PERF_MONITORING
    global_pd.ps.num_cases = iio->batch_size;
#endif

    for (batch_counter = 0; batch_counter < iio->batch_size; ++batch_counter) {
        (void) memcpy(runtime_data->mut_weights + runtime_data->input_offset,
                    iio->input + current_input_offset,
                    iio->in_len);

#ifdef ENABLE_PERF_MONITORING
        (void) resume_perf_monitoring(&global_pd);
#endif
        (runtime_data->inference_func)(runtime_data->const_weights,
                                    runtime_data->mut_weights,
                                    runtime_data->activations);
#ifdef ENABLE_PERF_MONITORING
        (void) pause_perf_monitoring(&global_pd);
        (void) read_perf_statistics(&global_pd);
        (void) memcpy(&(runtime_data->ps), &(global_pd.ps), sizeof(struct PerfStatistics));
#endif

        (void) memcpy(iio->output + current_output_offset,
                    runtime_data->mut_weights + runtime_data->output_offset,
                    iio->out_len);

        current_input_offset += iio->in_len;
        current_output_offset += iio->out_len;
    }
}

uint8_t 
*init_constant_weights(const char *wfname, const struct BundleConfig *bundle_config)
{
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
        size = (size_t)(file_offset);
        if (size != bundle_config->const_weight_vars_memsize) {
            fprintf(stderr,
                  "Unexpected file size (%zd) does not match expected (%llu)\n",
                  size, bundle_config->const_weight_vars_memsize);

            (void) close(fd);
            return NULL;
        }

        file_offset = lseek(fd, 0, SEEK_SET);
        if (file_offset == -1) {
            perror("Error processing weights file");

            (void) close(fd);
            return NULL;
        }

        retval = malloc_aligned(bundle_config->alignment, size);
        buffer = retval;
        if (retval != NULL) {
            while (bytes_total < size) {
                bytes_read = read(fd, buffer, size - bytes_total);
                bytes_total += bytes_read;

                if (bytes_read <= 0) {
                    if (bytes_read == -1)
                        perror("Error reading weights file");
                     else if (bytes_read == 0)
                        fprintf(stderr, "Error reading weights file: EOF reached too early\n");

                    free(retval);
                    retval = NULL;
                    break;
                }

                buffer += bytes_read;
            }
        }
        else
            perror("Error allocating memory for weights");
    }

    (void) close(fd);
    return retval;
}

uint8_t 
*init_mutable_weights(const struct BundleConfig *bundle_config)
{
    return malloc_aligned(bundle_config->alignment, bundle_config->mut_weight_vars_memsize);
}

uint8_t 
*init_activations(const struct BundleConfig *bundle_config)
{
    return malloc_aligned(bundle_config->alignment, bundle_config->activation_memsize);
}

void 
*malloc_aligned(uint64_t alignment, size_t size)
{
    int result = 0;
    void *retval = NULL;

    result = posix_memalign(&retval, alignment, size);
    if (result != 0) {
        fprintf(stderr, "Error allocating memory (%d): %s\n", result, strerror(result));
        retval = NULL;
    } else {
        (void) memset(retval, 0x0, size);
    }

    return retval;
}

int 
get_io_offsets(const struct NetworkData *network_data, size_t *in_offset, size_t *out_offset)
{
    int retval;
    bool found_in = false;
    bool found_out = false;
    size_t symbol_index;
    size_t input_name_len;
    size_t output_name_len;

    input_name_len = strlen(network_data->input_tensor_name);
    output_name_len = strlen(network_data->output_tensor_name);

    for (symbol_index = 0; symbol_index < network_data->bundle_config->num_symbols; ++symbol_index) {
        if (strncmp(network_data->input_tensor_name, network_data->bundle_config->symbols[symbol_index].name, input_name_len) == 0) {
            *in_offset = network_data->bundle_config->symbols[symbol_index].offset;
            found_in = true;

            if (found_out)
                break;
        }
        if (strncmp(network_data->output_tensor_name, network_data->bundle_config->symbols[symbol_index].name, output_name_len) == 0) {
            *out_offset = network_data->bundle_config->symbols[symbol_index].offset;
            found_out = true;

            if (found_in)
                break;
        }
    }

    if (!found_in || !found_out)
        retval = X_FAILURE;
    else
        retval = X_SUCCESS;

    return retval;
}
