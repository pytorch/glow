// Copyright 2017 Facebook Inc.  All Rights Reserved.

#include "glow/Base/Image.h"
#include "glow/Base/Tensor.h"
#include "glow/Support/Support.h"

using namespace glow;

#if WITH_PNG
#include <png.h>

bool glow::readPngImage(Tensor *T, const char *filename,
                        std::pair<float, float> range) {
  unsigned char header[8];
  // open file and test for it being a png.
  FILE *fp = fopen(filename, "rb");
  // Can't open the file.
  if (!fp) {
    return true;
  }

  // Validate signature.
  fread(header, 1, 8, fp);
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
  (void)bit_depth;
  assert(bit_depth == 8 && "Invalid image");
  assert((color_type == PNG_COLOR_TYPE_RGB_ALPHA ||
          color_type == PNG_COLOR_TYPE_RGB) &&
         "Invalid image");
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

  T->reset(ElemKind::FloatTy, {width, height, 3});
  auto H = T->getHandle<>();

  float scale = ((range.second - range.first) / 255.0);
  float bias = range.first;

  for (size_t row_n = 0; row_n < height; row_n++) {
    png_byte *row = row_pointers[row_n];
    for (size_t col_n = 0; col_n < width; col_n++) {
      png_byte *ptr = &(row[col_n * (hasAlpha ? 4 : 3)]);

      H.at({row_n, col_n, 0}) = float(ptr[0]) * scale + bias;
      H.at({row_n, col_n, 1}) = float(ptr[1]) * scale + bias;
      H.at({row_n, col_n, 2}) = float(ptr[2]) * scale + bias;
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

bool glow::writePngImage(Tensor *T, const char *filename,
                         std::pair<float, float> range) {
  /* create file */
  FILE *fp = fopen(filename, "wb");
  if (!fp) {
    return true;
  }

  /* initialize stuff */
  png_structp png_ptr =
      png_create_write_struct(PNG_LIBPNG_VER_STRING, nullptr, nullptr, nullptr);

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

  if (setjmp(png_jmpbuf(png_ptr))) {
    return true;
  }

  auto H = T->getHandle<>();

  auto odim = H.dims();
  assert(odim[2] < 4 && "Invalid buffer to save");

  size_t width = odim[0];
  size_t height = odim[1];
  int color_type = PNG_COLOR_TYPE_RGB_ALPHA;
  int bit_depth = 8;

  png_set_IHDR(png_ptr, info_ptr, width, height, bit_depth, color_type,
               PNG_INTERLACE_NONE, PNG_COMPRESSION_TYPE_BASE,
               PNG_FILTER_TYPE_BASE);

  png_write_info(png_ptr, info_ptr);

  if (setjmp(png_jmpbuf(png_ptr))) {
    return true;
  }

  auto *row_pointers = (png_bytep *)malloc(sizeof(png_bytep) * height);
  for (size_t y = 0; y < height; y++) {
    row_pointers[y] = (png_byte *)malloc(png_get_rowbytes(png_ptr, info_ptr));
  }

  float scale = ((range.second - range.first) / 255.0);
  float bias = range.first;

  for (size_t y = 0; y < height; y++) {
    png_byte *row = row_pointers[y];
    for (size_t x = 0; x < width; x++) {
      png_byte *ptr = &(row[x * 4]);
      ptr[0] = H.at({x, y, 0}) * scale + bias;
      ptr[1] = H.at({x, y, 1}) * scale + bias;
      ptr[2] = H.at({x, y, 2}) * scale + bias;
      ptr[3] = 0xff;
    }
  }

  png_write_image(png_ptr, row_pointers);

  if (setjmp(png_jmpbuf(png_ptr))) {
    return true;
  }

  png_write_end(png_ptr, nullptr);

  /* cleanup heap allocation */
  for (size_t y = 0; y < height; y++) {
    free(row_pointers[y]);
  }
  free(row_pointers);
  png_destroy_read_struct(&png_ptr, &info_ptr, (png_infopp)NULL);
  fclose(fp);
  return false;
}

#else
bool glow::readPngImage(Tensor *T, const char *filename,
                        std::pair<float, float> range) {
  GLOW_ASSERT(false && "Not configured with libpng");
}

bool glow::writePngImage(Tensor *T, const char *filename,
                         std::pair<float, float> range) {
  GLOW_ASSERT(false && "Not configured with libpng");
}
#endif
