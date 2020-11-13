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
#include <png.h>
#include <stdio.h>
#include <stdlib.h>

#include <cstring>
#include <string>
#include <vector>

#include "lenet_mnist.h"

/// This is an example demonstrating how to use auto-generated bundles and
/// create standalone executables that can perform neural network computations.
/// This example loads and runs the compiled lenet_mnist network model.
/// This example is using the static bundle API.

#define DEFAULT_HEIGHT 28
#define DEFAULT_WIDTH 28
#define OUTPUT_LEN 10

//===----------------------------------------------------------------------===//
//                   Image processing helpers
//===----------------------------------------------------------------------===//
std::vector<std::string> inputImageFilenames;

/// \returns the index of the element at x,y,z,w.
size_t getXYZW(const size_t *dims, size_t x, size_t y, size_t z, size_t w) {
  return (x * dims[1] * dims[2] * dims[3]) + (y * dims[2] * dims[3]) +
         (z * dims[3]) + w;
}

/// \returns the index of the element at x,y,z.
size_t getXYZ(const size_t *dims, size_t x, size_t y, size_t z) {
  return (x * dims[1] * dims[2]) + (y * dims[2]) + z;
}

/// Reads a PNG image from a file into a newly allocated memory block \p imageT
/// representing a WxHxNxC tensor and returns it. The client is responsible for
/// freeing the memory block.
bool readPngImage(const char *filename, std::pair<float, float> range,
                  float *&imageT, size_t *imageDims) {
  unsigned char header[8];
  // open file and test for it being a png.
  FILE *fp = fopen(filename, "rb");
  // Can't open the file.
  if (!fp) {
    return true;
  }

  // Validate signature.
  size_t fread_ret = fread(header, 1, 8, fp);
  if (fread_ret != 8) {
    return true;
  }
  if (png_sig_cmp(header, 0, 8)) {
    return true;
  }

  // Initialize stuff.
  png_structp png_ptr =
      png_create_read_struct(PNG_LIBPNG_VER_STRING, nullptr, nullptr, nullptr);
  if (!png_ptr) {
    return true;
  }

  png_infop info_ptr = png_create_info_struct(png_ptr);
  if (!info_ptr) {
    return true;
  }

  if (setjmp(png_jmpbuf(png_ptr))) {
    return true;
  }

  png_init_io(png_ptr, fp);
  png_set_sig_bytes(png_ptr, 8);
  png_read_info(png_ptr, info_ptr);

  size_t width = png_get_image_width(png_ptr, info_ptr);
  size_t height = png_get_image_height(png_ptr, info_ptr);
  int color_type = png_get_color_type(png_ptr, info_ptr);
  int bit_depth = png_get_bit_depth(png_ptr, info_ptr);

  const bool isGray = color_type == PNG_COLOR_TYPE_GRAY;
  const size_t numChannels = 1;

  (void)bit_depth;
  assert(bit_depth == 8 && "Invalid image");
  assert(isGray && "Invalid image");
  (void)isGray;
  bool hasAlpha = (color_type == PNG_COLOR_TYPE_RGB_ALPHA);

  int number_of_passes = png_set_interlace_handling(png_ptr);
  (void)number_of_passes;
  assert(number_of_passes == 1 && "Invalid image");

  png_read_update_info(png_ptr, info_ptr);

  // Error during image read.
  if (setjmp(png_jmpbuf(png_ptr))) {
    return true;
  }

  auto *row_pointers = (png_bytep *)malloc(sizeof(png_bytep) * height);
  for (size_t y = 0; y < height; y++) {
    row_pointers[y] = (png_byte *)malloc(png_get_rowbytes(png_ptr, info_ptr));
  }

  png_read_image(png_ptr, row_pointers);
  png_read_end(png_ptr, info_ptr);

  imageDims[0] = width;
  imageDims[1] = height;
  imageDims[2] = numChannels;
  imageT = static_cast<float *>(
      calloc(1, width * height * numChannels * sizeof(float)));

  float scale = ((range.second - range.first) / 255.0);
  float bias = range.first;

  for (size_t row_n = 0; row_n < height; row_n++) {
    png_byte *row = row_pointers[row_n];
    for (size_t col_n = 0; col_n < width; col_n++) {
      png_byte *ptr =
          &(row[col_n * (hasAlpha ? (numChannels + 1) : numChannels)]);
      imageT[getXYZ(imageDims, row_n, col_n, 0)] = float(ptr[0]) * scale + bias;
    }
  }

  for (size_t y = 0; y < height; y++) {
    free(row_pointers[y]);
  }
  free(row_pointers);
  png_destroy_read_struct(&png_ptr, &info_ptr, (png_infopp)NULL);
  fclose(fp);
  return false;
}

/// Loads and normalizes all PNGs into a tensor memory block \p resultT in the
/// NCHW 1x28x28 format.
static void loadImagesAndPreprocess(const std::vector<std::string> &filenames,
                                    float *&resultT, size_t *resultDims) {
  assert(filenames.size() > 0 &&
         "There must be at least one filename in filenames");
  std::pair<float, float> range = std::make_pair(0., 1.0);
  unsigned numImages = filenames.size();
  // N x C x H x W
  resultDims[0] = numImages;
  resultDims[1] = 1;
  resultDims[2] = DEFAULT_HEIGHT;
  resultDims[3] = DEFAULT_WIDTH;
  size_t resultSizeInBytes =
      numImages * DEFAULT_HEIGHT * DEFAULT_WIDTH * sizeof(float);
  resultT = static_cast<float *>(malloc(resultSizeInBytes));
  // We iterate over all the png files, reading them all into our result tensor
  // for processing
  for (unsigned n = 0; n < numImages; n++) {
    float *imageT{nullptr};
    size_t dims[3];
    bool loadSuccess = !readPngImage(filenames[n].c_str(), range, imageT, dims);
    assert(loadSuccess && "Error reading input image.");
    (void)loadSuccess;

    assert((dims[0] == DEFAULT_HEIGHT && dims[1] == DEFAULT_WIDTH) &&
           "All images must have the same Height and Width");

    // Convert to BGR, as this is what NN is expecting.
    for (unsigned y = 0; y < dims[1]; y++) {
      for (unsigned x = 0; x < dims[0]; x++) {
        resultT[getXYZW(resultDims, n, 0, x, y)] =
            imageT[getXYZ(dims, x, y, 0)];
      }
    }
  }
}

//===----------------------------------------------------------------------===//
//                 Wrapper code for executing a bundle
//===----------------------------------------------------------------------===//
/// Statically allocate memory for constant weights (model weights) and
/// initialize.
GLOW_MEM_ALIGN(LENET_MNIST_MEM_ALIGN)
uint8_t constantWeight[LENET_MNIST_CONSTANT_MEM_SIZE] = {
#include "lenet_mnist.weights.txt"
};

/// Statically allocate memory for mutable weights (model input/output data).
GLOW_MEM_ALIGN(LENET_MNIST_MEM_ALIGN)
uint8_t mutableWeight[LENET_MNIST_MUTABLE_MEM_SIZE];

/// Statically allocate memory for activations (model intermediate results).
GLOW_MEM_ALIGN(LENET_MNIST_MEM_ALIGN)
uint8_t activations[LENET_MNIST_ACTIVATIONS_MEM_SIZE];

/// Bundle input data absolute address.
uint8_t *inputAddr = GLOW_GET_ADDR(mutableWeight, LENET_MNIST_data);

/// Bundle output data absolute address.
uint8_t *outputAddr = GLOW_GET_ADDR(mutableWeight, LENET_MNIST_softmax);

/// Copy the pre-processed images into the mutable region of the bundle.
static void initInputImages() {
  size_t inputDims[4];
  float *inputT{nullptr};
  loadImagesAndPreprocess(inputImageFilenames, inputT, inputDims);
  // Copy image data into the data input variable in the mutableWeightVars area.
  size_t imageDataSizeInBytes =
      inputDims[0] * inputDims[1] * inputDims[2] * inputDims[3] * sizeof(float);
  memcpy(inputAddr, inputT, imageDataSizeInBytes);
  free(inputT);
}

/// Dump the result of the inference by looking at the results vector and
/// finding the index of the max element.
static void printResults() {
  int maxIdx = 0;
  float maxValue = 0;
  float *results = (float *)(outputAddr);
  for (int i = 0; i < OUTPUT_LEN; ++i) {
    if (results[i] > maxValue) {
      maxValue = results[i];
      maxIdx = i;
    }
  }
  printf("Result: %u\n", maxIdx);
  printf("Confidence: %f\n", maxValue);
}

//===----------------------------------------------------------------------===//
//                           Instrumentation callbacks
//===----------------------------------------------------------------------===//
/// Dump data to file.
void dumpToFile(void *data, int len, char *fileName) {
  FILE *fh;
  fh = fopen(fileName, "wb");
  if (!fh) {
    printf("ERROR: Error opening file!");
  }
  size_t lenWrite = fwrite(data, 1, len, fh);
  if (len != lenWrite) {
    printf("ERROR: Error writing to file!");
  }
  fclose(fh);
}

#ifndef INSTRUMENT_IR_DIR
#error "INSTRUMENT_IR_DIR not defined"
#endif

// Variables used to store the dump file names.
char filename[1000];
int buffIdx = 0;

// Variables to check the number of instructions and dumps.
int instrNum = 0;
int buffNum = 0;

extern "C" {
// We dump the same binary content as the -instrument-debug which dumps
// binaries with names dataxxxx.bin starting with the output operands and then
// the input operands.
void glow_instrument_before(int id, int kind, int opInp, int opOut,
                            uint8_t **opAddr, int *opSize) {
  // Check instruction ID.
  if (id != instrNum) {
    printf("ERROR: Instruction ID mismatch!\n");
  }
  // Dump the input operands.
  buffIdx += opOut;
  for (size_t idx = 0; idx < opInp; idx++) {
    sprintf(filename, "%s/data%04d.bin", INSTRUMENT_IR_DIR, buffIdx);
    buffIdx++;
    dumpToFile(opAddr[idx], opSize[idx], filename);
    buffNum++;
  }
  buffIdx -= opInp;
  buffIdx -= opOut;
}

void glow_instrument_after(int id, int kind, int opInp, int opOut,
                           uint8_t **opAddr, int *opSize) {
  // Check instruction ID.
  if (id != instrNum) {
    printf("ERROR: Instruction ID mismatch!\n");
  }
  instrNum++;
  // Dump the output operands.
  for (size_t idx = 0; idx < opOut; idx++) {
    sprintf(filename, "%s/data%04d.bin", INSTRUMENT_IR_DIR, buffIdx);
    buffIdx++;
    dumpToFile(opAddr[opInp + idx], opSize[opInp + idx], filename);
    buffNum++;
  }
  buffIdx += opInp;
}
}

int main(int argc, char **argv) {

  // First argument is the image used for inference.
  if (argc != 2) {
    printf("ERROR: Invalid number of arguments!\n");
    return 1;
  }
  inputImageFilenames.push_back(argv[1]);

  // Initialize input images.
  initInputImages();

  // Perform the computation.
  int errCode = lenet_mnist(constantWeight, mutableWeight, activations);
  if (errCode != GLOW_SUCCESS) {
    printf("Error running bundle: error code %d\n", errCode);
  }

  // Check number of dumps.
  printf("Number of instructions: %d\n", instrNum);
  printf("Number of data dumps: %d\n", buffNum);

  // Print results.
  printResults();
}
