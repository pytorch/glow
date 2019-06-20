#include <stdio.h>
#include <argp.h>
#include <fcntl.h>
#include <string.h>
#include <unistd.h>
#include <dlfcn.h>
#include <assert.h>

#include "x_inference_lib.h"

#ifndef X_USE_DYNAMIC

#ifndef X_MODEL_NAME
#error "X_MODEL_NAME must be defined when using static bundles!"
#endif

#define GLUE(m, c) m ## _ ## c
#define BUILD_SYMBOL_NAME(m, c) GLUE(m, c)

extern struct BundleConfig BUILD_SYMBOL_NAME(X_MODEL_NAME, config);
extern void X_MODEL_NAME(uint8_t *, uint8_t *, uint8_t *);

#endif

const char *argp_program_version = "x-infer v0.01";
const char *argp_program_bug_address = "<william.yessen@xperi.com>";
const char doc[] = 
                   "\n                    Generic Inference Engine                         \n"
                   "-----------------------------------------------------------------------\n"
#ifdef X_USE_DYNAMIC
                   "Dynamic bundle loading: SUPPORTED\n"
#else
                   "Dynamic bundle loading: NOT SUPPORTED (bundle has been statically linked)\n"
#endif
#ifdef ENABLE_PERF_MONITORING
                   "Performance monitoring: SUPPORTED\n"
#else
                   "Performance monitoring: NOT SUPPORTED\n"
#endif
                   "\nx-infer runs inference against the provided Glow bundle. Weights file must be "
                   "specified as the first argument. The input file must be specified with "
                   "[--infile] (binary file). Input tensor type [-intype], output tensor type [-outtype], "
                   "input length [--inlen], output length [--outlen], input tensor name [-inname], "
                   "and output tensor name [--outname] must be specified. "
                   "\n\n"
                   "When built with dynamic bundle loading support, bundle must be specified as the "
                   "first positional argument, and the weights file as the second. When built with "
                   "a bundle statically linked in, dynamic loading is not supported "
#ifdef ENABLE_PERF_MONITORING
                   "\n\nAdditionally, performance monitoring is available with [--perf] option. In this case, "
                   "[--perflog] can be used to specify the performance log filename. Otherwise, performance log "
                   "is written to stdout "
#endif

                   "\n\nShort and long form options are: ";
const char args_doc[] =
#ifdef X_USE_DYNAMIC
"[BUNDLE FILENAME] "
#endif
"[WEIGHTS FILENAME]";

const struct argp_option options[] = {
  {"output",   'o', "FILE",   0,  "Output to binary FILE instead of standard output" },
  {"infile",    'i', "FILE",   0,  "Input from FILE" },
  {"intype",   't', "TYPE",   0,  "Input TYPE (one of F32, F16, I16, I8)" },
  {"outtype",  'T', "TYPE",   0,  "Output TYPE (one of F32, F16, I16, I8)" },
  {"inlen",    'l', "LEN",    0,  "Input tensor length (e.g. if the tensor is of shape 2x3x4, its "
                               "length is 2 * 3 * 4 = 24)"},
  {"outlen",   'L', "LEN",    0,  "Output tensor length (e.g. if the tensor is of shape 2x3x4, its "
                               "length is 2 * 3 * 4 = 24)"},
  {"inname",   'n', "NAME",   0,  "Input tensor name NAME" },
  {"outname",  'N', "NAME",   0,  "Output tensor name NAME" },

#ifdef ENABLE_PERF_MONITORING
  {"perf",     'p', 0,        0,  "Whether to output performance logs" },
  {"perflog",   'P', "NAME",   0,  "Performance log output filename (stdout if omitted)" },
#endif
  
  #ifdef X_USE_DYNAMIC
  {"model",    'm', "NAME",   0,  "Model name (maximum 128 chars)" },
  #endif

  { 0 }
};

struct arguments
{
#ifdef X_USE_DYNAMIC
    char *bundle_file;
    char bundle_config_name[135];
    char *model_name;
#endif
#ifdef ENABLE_PERF_MONITORING
    int perf_monitor;
    char *perf_log_file;
#endif
    char *weights_file;
    char *out_file;
    char *in_file;
    char *in_type;
    char *out_type;
    char *in_len;
    char *out_len;
    char *in_name;
    char *out_name;
    size_t in_tensor_size;
    size_t out_tensor_size;
    size_t batch;
};

static error_t parse_opt(int key, char *arg, struct argp_state *state);
static void init_arguments(struct arguments *arguments);
static int compute_arguments(struct arguments *arguments);
static void cleanup_arguments(struct arguments *arguments);
static int retreive_and_load_input(struct InferenceIO *inference_io, const struct arguments *arguments);
static int retreive_and_store_output(struct InferenceIO *inference_io, const struct arguments *arguments);

#ifdef ENABLE_PERF_MONITORING
static void report_performance(const struct PerfStatistics *ps, const char *filename);
#endif

static struct argp argp = { options, parse_opt, args_doc, doc };

int main(int argc, char **argv)
{
    struct arguments arguments;
    struct NetworkData network_data;
    struct RuntimeData runtime_data;
    struct InferenceIO inference_io;
    int retval;

#ifdef X_USE_DYNAMIC
    struct BundleConfig *x_config;
    InferenceFunctionPtr_t x_infer;
    void *dl_handle;
#endif

    init_arguments(&arguments);
    retval = argp_parse(&argp, argc, argv, 0, 0, &arguments);
    if (retval != X_SUCCESS) {
        fprintf(stderr, "ERROR: Invalid arguments. Use --help for help.");
        cleanup_arguments(&arguments);
        exit(retval);
    }

    retval = compute_arguments(&arguments);
    if (retval != X_SUCCESS) {
        cleanup_arguments(&arguments);
        exit(retval);
    }

#ifdef X_USE_DYNAMIC
    /* Load the bundle... */
    dl_handle = dlopen(arguments.bundle_file, RTLD_NOW);
    if (dl_handle == NULL) {
        fprintf(stderr, "ERROR: Cannot load bundle %s: %s\n", arguments.bundle_file, dlerror());
        cleanup_arguments(&arguments);
        exit(X_FAILURE);
    }

    x_config = dlsym(dl_handle, arguments.bundle_config_name);
    if (dlerror() != NULL) {
        fprintf(stderr, "ERROR: Cannot load bundle config structure %s from %s: %s.\n", 
                        arguments.bundle_config_name, arguments.bundle_file, dlerror());
        cleanup_arguments(&arguments);
        exit(X_FAILURE);
    }

    x_infer = dlsym(dl_handle, arguments.model_name);
    if (dlerror() != NULL) {
        fprintf(stderr, "ERROR: Cannot load model %s from %s: %s.\n", 
                        arguments.model_name, arguments.bundle_file, dlerror());
        cleanup_arguments(&arguments);
        exit(X_FAILURE);
    }
#endif

    /* Load the network data... */
#ifdef X_USE_DYNAMIC
    network_data.bundle_config = x_config;
    network_data.inference_function = x_infer;
#else
    network_data.bundle_config = &BUILD_SYMBOL_NAME(X_MODEL_NAME, config);
    network_data.inference_function = X_MODEL_NAME;
#endif

    network_data.input_tensor_name = arguments.in_name;
    network_data.output_tensor_name = arguments.out_name;
    network_data.weights_file_name = arguments.weights_file;

    /* Initialize the runtime data... */
    retval = init_runtime_data(&network_data, &runtime_data);
    if (retval != X_SUCCESS) {
        fprintf(stderr, "ERROR: Could not initialize data. Exiting!\n");
        cleanup_runtime_data(&runtime_data);
        cleanup_arguments(&arguments);
        exit(retval);
    }

    /* Initialize the IO struct... */
    inference_io.in_len = arguments.in_tensor_size;
    inference_io.out_len = arguments.out_tensor_size;
    inference_io.batch_size = arguments.batch;

    retval = init_io(&inference_io, NULL, NULL);
    if (retval != X_SUCCESS) {
        fprintf(stderr, "ERROR: Could not initialize IO. Exiting!\n");
        cleanup_runtime_data(&runtime_data);
        cleanup_io(&inference_io);
        cleanup_arguments(&arguments);
        exit(retval);
    }

    retval = retreive_and_load_input(&inference_io, &arguments);
    if (retval != X_SUCCESS) {
        cleanup_runtime_data(&runtime_data);
        cleanup_io(&inference_io);
        cleanup_arguments(&arguments);
        exit (retval);
    }

    run_inference(&inference_io, &runtime_data);

    retval = retreive_and_store_output(&inference_io, &arguments);
    if (retval != X_SUCCESS) {
        cleanup_runtime_data(&runtime_data);
        cleanup_io(&inference_io);
        cleanup_arguments(&arguments);
        exit (retval);
    }

#ifdef ENABLE_PERF_MONITORING
    report_performance(&(runtime_data.ps), arguments.perf_log_file);
#endif

    /* Finally, don't forget to clean up... */
    cleanup_runtime_data(&runtime_data);
    cleanup_io(&inference_io);
    cleanup_arguments(&arguments);

    exit(0);
}

error_t
parse_opt (int key, char *arg, struct argp_state *state)
{
  struct arguments *arguments = state->input;

    switch (key) {
        case 'o':
            arguments->out_file = arg;
            break;
        case 'i':
            arguments->in_file = arg;
            break;
        case 't':
            arguments->in_type = arg;
            break;
        case 'T':
            arguments->out_type = arg;
            break;
        case 'l':
            arguments->in_len = arg;
            break;
        case 'L':
            arguments->out_len = arg;
            break;
        case 'n':
            arguments->in_name = arg;
            break;
        case 'N':
            arguments->out_name = arg;
            break;
#ifdef ENABLE_PERF_MONITORING
        case 'p':
            arguments->perf_monitor = 1;
            break;
        case 'P':
            arguments->perf_log_file = arg;
            break;
#endif
#ifdef X_USE_DYNAMIC
        case 'm':
            arguments->model_name = arg;
            break;
#endif
        case ARGP_KEY_ARG:
#ifdef X_USE_DYNAMIC
            if (state->arg_num >= 3)
                argp_usage(state);
            if (state->arg_num == 1)
                arguments->weights_file = arg;
            else
                arguments->bundle_file = arg;
#else
            if (state->arg_num >= 2)
                argp_usage(state);
            arguments->weights_file = arg;
#endif
            break;
        case ARGP_KEY_END:
#ifdef X_USE_DYNAMIC
            if (state->arg_num < 2)
#else
            if (state->arg_num < 1)
#endif
                argp_usage (state);
            break;
        default:
            return ARGP_ERR_UNKNOWN;
    }

    return X_SUCCESS;
}

void init_arguments(struct arguments *arguments)
{
    memset(arguments, 0x0, sizeof(struct arguments));
}

int compute_arguments(struct arguments *arguments)
{
    int fd;
    off_t file_offset;
    size_t input_size;
    size_t in_tensor_type_size;
    size_t out_tensor_type_size;
    long in_tensor_len;
    long out_tensor_len;

#ifdef X_USE_DYNAMIC
    if (arguments->model_name == NULL) {
        fprintf(stderr, "ERROR: -m option must be specified.\n");
        return X_FAILURE;
    }
    sprintf(arguments->bundle_config_name, "%s_%s", arguments->model_name, "config");
#endif

    if (arguments->in_type == NULL || arguments->out_type == NULL) {
        fprintf(stderr, "ERROR: -t and -T must be specified.\n");
        return X_FAILURE;
    }

    if (strncmp(arguments->in_type, "F32", 4) == 0)
        in_tensor_type_size = 4;
    else if (strncmp(arguments->in_type, "F16", 4) == 0 ||
             strncmp(arguments->in_type, "I16", 4) == 0)
        in_tensor_type_size = 2;
    else if (strncmp(arguments->in_type, "I8", 3) == 0)
        in_tensor_type_size = 1;
    else {
        fprintf(stderr, "ERROR: Invalid input tensor type %s\n", arguments->in_type);
        return X_FAILURE;
    }

    if (strncmp(arguments->out_type, "F32", 4) == 0)
        out_tensor_type_size = 4;
    else if (strncmp(arguments->out_type, "F16", 4) == 0 ||
             strncmp(arguments->out_type, "I16", 4) == 0)
        out_tensor_type_size = 2;
    else if (strncmp(arguments->out_type, "I8", 3) == 0)
        out_tensor_type_size = 1;
    else {
        fprintf(stderr, "ERROR: Invalid output tensor type %s\n", arguments->out_type);
        return X_FAILURE;
    }

    if (arguments->in_len == NULL || arguments->out_len == NULL) {
        fprintf(stderr, "ERROR: -l and -L options must be specified.\n");
        return X_FAILURE;
    }

    in_tensor_len = atol(arguments->in_len);
    out_tensor_len = atol(arguments->out_len);
    if (in_tensor_len <= 0) {
        fprintf(stderr, "ERROR: Invalid -l value: %s.\n", arguments->in_len);
        return X_FAILURE;
    }
    if (out_tensor_len <= 0) {
        fprintf(stderr, "ERROR: Invalid -L value: %s.\n", arguments->out_len);
        return X_FAILURE;
    }

    arguments->in_tensor_size = (size_t)(in_tensor_len) * in_tensor_type_size;
    arguments->out_tensor_size = (size_t)(out_tensor_len) * out_tensor_type_size;

    if (arguments->in_name == NULL || arguments->out_name == NULL) {
        fprintf(stderr, "ERROR: both -n and -N options must be specified.\n");
        return X_FAILURE;
    }

    if (arguments->in_file == NULL) {
        fprintf(stderr, "ERROR: -i option must be specified.\n");
        return X_FAILURE;
    }

    if (arguments->out_file == NULL) {
        fprintf(stderr, "ERROR: -o option must be specified.\n");
        return X_FAILURE;
    }

    fd = open(arguments->in_file, O_RDONLY);
    if (fd == -1) {
        perror("ERROR: Could not process input file");
        return X_FAILURE;
    }

    file_offset = lseek(fd, 0, SEEK_END);
    if (file_offset == -1) {
        perror("ERRPR: Could not process input file");
        (void) close(fd);
        return X_FAILURE;
    }
    input_size = (size_t)(file_offset);

    if (input_size % arguments->in_tensor_size != 0) {
        fprintf(stderr, "ERROR: Input file size (%zu bytes) is not a multiple of input tensor size (%zu bytes).\n",
                input_size, arguments->in_tensor_size);
        return X_FAILURE;
    }

    arguments->batch = input_size / arguments->in_tensor_size;

    return X_SUCCESS;
}

void cleanup_arguments(struct arguments *arguments)
{
    init_arguments(arguments);
}

int retreive_and_load_input(struct InferenceIO *inference_io, const struct arguments *arguments)
{
    const size_t size = arguments->batch * arguments->in_tensor_size;
    size_t bytes_total;
    int bytes_read;
    int fd;
    int retval = X_SUCCESS;

    fd = open(arguments->in_file, O_RDONLY);
    if (fd == -1) {
        perror("ERROR: Could not process input file");
        return X_FAILURE;
    }

    bytes_total = 0;
    while (bytes_total < size) {
        bytes_read = read(fd, inference_io->input, size - bytes_total);
        bytes_total += bytes_read;

        if (bytes_read <= 0) {
            if (bytes_read == -1)
                perror("ERROR: Could not read input file");
            else if (bytes_read == 0)
                fprintf(stderr, "ERROR: Could not read input file - EOF reached too early.\n");

            retval = X_FAILURE;
            break;
        }
    }
    (void) close(fd);

    return retval;
}

int retreive_and_store_output(struct InferenceIO *inference_io, const struct arguments *arguments)
{
    const size_t size = arguments->batch * arguments->out_tensor_size;
    size_t bytes_total;
    int bytes_written;
    int fd;
    int retval = X_SUCCESS;

    fd = open(arguments->out_file, O_CREAT | O_WRONLY, S_IRUSR | S_IWUSR);
    if (fd == -1) {
        perror("ERROR: Could not open output file");
        return X_FAILURE;
    }

    bytes_total = 0;
    while (bytes_total < size) {
        bytes_written = write(fd, inference_io->output, size - bytes_total);
        bytes_total += bytes_written;

        if (bytes_written <= 0) {
            if (bytes_written == -1) {
                perror("ERROR: Could not write to output file");
                retval = X_FAILURE;
                break;
            }
            else if (bytes_written == 0)
                fprintf(stderr, "WARNING: Wrote 0 bytes (is device busy? will retry).\n");
        }
    }
    (void) close(fd);

    return retval;
}

#ifdef ENABLE_PERF_MONITORING
void report_performance(const struct PerfStatistics *ps, const char *filename)
{
    int fd;
    const size_t output_buffer_size = 512;
    char buffer[output_buffer_size] = { 0 };
    int bytes_written;
    size_t bytes_total;

    snprintf(buffer, output_buffer_size, "\nConstant weights size       : %zd bytes\n"
                                         "Number of cases             : %zd\n"
                                         "Number of CPU cycles (x1-e6): %f\n\n",
                                         ps->const_weights_size,
                                         ps->num_cases,
                                         ps->num_cpu_cycles/1.0e6);

    if (filename != NULL) {
        fd = open(filename, O_CREAT | O_WRONLY, S_IRUSR | S_IWUSR);
        if (fd == -1) {
            perror("ERROR: Could not open perf log file for writing");
            return;
        }
    } else {
        fd = STDOUT_FILENO;
    }

    bytes_total = 0;
    while(bytes_total < output_buffer_size) {
        bytes_written = write(fd, buffer, output_buffer_size - bytes_total);
        bytes_total += bytes_written;

        if (bytes_written <= 0) {
            if (bytes_written == -1) {
                perror("ERROR: Could not write to perf log file");
                break;
            } else if (bytes_written == 0) {
                fprintf(stderr, "WARNING: Wrote 0 bytes (is device busy? will retry).\n");
            }
        }

    }
    
    if (fd != STDOUT_FILENO)
        (void) close(fd);
}
#endif