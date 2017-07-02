#ifndef NOETHER_IMAGE_H
#define NOETHER_IMAGE_H

#include "noether/Layer.h"
#include "noether/Tensor.h"

#include <cassert>
#include <cstddef>
#include <cstdint>
#include <string>

#include <png.h>

template <class ElemTy> class PNGLayer final : public Layer<ElemTy> {
public:
  PNGLayer() {}

  virtual std::string getName() const override { return "PNGLayer"; }

  /// Reads a png image. \returns True if an error occurred.
  bool readImage(const char *filename) {
    unsigned char header[8];
    // open file and test for it being a png.
    FILE *fp = fopen(filename, "rb");
    // Can't open the file.
    if (!fp)
      return true;

    // Validate signature.
    fread(header, 1, 8, fp);
    if (png_sig_cmp(header, 0, 8))
      return true;

    // Initialize stuff.
    png_structp png_ptr =
        png_create_read_struct(PNG_LIBPNG_VER_STRING, NULL, NULL, NULL);
    if (!png_ptr)
      return true;

    png_infop info_ptr = png_create_info_struct(png_ptr);
    if (!info_ptr)
      return true;

    if (setjmp(png_jmpbuf(png_ptr)))
      return true;

    png_init_io(png_ptr, fp);
    png_set_sig_bytes(png_ptr, 8);
    png_read_info(png_ptr, info_ptr);

    int width = png_get_image_width(png_ptr, info_ptr);
    int height = png_get_image_height(png_ptr, info_ptr);
    int color_type = png_get_color_type(png_ptr, info_ptr);
    int bit_depth = png_get_bit_depth(png_ptr, info_ptr);
    assert(bit_depth == 8 && "Invalid image");
    assert(color_type == PNG_COLOR_TYPE_RGB_ALPHA ||
           color_type == PNG_COLOR_TYPE_RGB && "Invalid image");
    bool hasAlpha = (color_type == PNG_COLOR_TYPE_RGB_ALPHA);

    int number_of_passes = png_set_interlace_handling(png_ptr);
    assert(number_of_passes == 1 && "Invalid image");

    png_read_update_info(png_ptr, info_ptr);

    // Error during image read.
    if (setjmp(png_jmpbuf(png_ptr)))
      return true;

    png_bytep *row_pointers = (png_bytep *)malloc(sizeof(png_bytep) * height);
    for (int y = 0; y < height; y++)
      row_pointers[y] = (png_byte *)malloc(png_get_rowbytes(png_ptr, info_ptr));

    png_read_image(png_ptr, row_pointers);
    fclose(fp);

    this->output_.weight_.reset(width, height, 3);

    for (int y = 0; y < height; y++) {
      png_byte *row = row_pointers[y];
      for (int x = 0; x < width; x++) {
        png_byte *ptr = &(row[x * (hasAlpha ? 4 : 3)]);

        this->output_.weight_.get(x, y, 0) = ptr[0];
        this->output_.weight_.get(x, y, 1) = ptr[1];
        this->output_.weight_.get(x, y, 2) = ptr[2];
      }
    }

    for (int y = 0; y < height; y++)
      free(row_pointers[y]);
    free(row_pointers);

    return false;
  }

  bool writeImage(const char *filename) {
    /* create file */
    FILE *fp = fopen(filename, "wb");
    if (!fp)
      return true;

    /* initialize stuff */
    png_structp png_ptr =
        png_create_write_struct(PNG_LIBPNG_VER_STRING, NULL, NULL, NULL);

    if (!png_ptr)
      return true;

    png_infop info_ptr = png_create_info_struct(png_ptr);
    if (!info_ptr)
      return true;

    if (setjmp(png_jmpbuf(png_ptr)))
      return true;

    png_init_io(png_ptr, fp);

    if (setjmp(png_jmpbuf(png_ptr)))
      return true;

    size_t ix, iy, iz;
    std::tie(ix, iy, iz) = this->output_.dims();
    assert(iz < 4 && "Invalid buffer to save");

    int width = ix;
    int height = iy;
    int color_type = PNG_COLOR_TYPE_RGB_ALPHA;
    int bit_depth = 8;

    png_set_IHDR(png_ptr, info_ptr, width, height, bit_depth, color_type,
                 PNG_INTERLACE_NONE, PNG_COMPRESSION_TYPE_BASE,
                 PNG_FILTER_TYPE_BASE);

    png_write_info(png_ptr, info_ptr);

    if (setjmp(png_jmpbuf(png_ptr)))
      return true;

    png_bytep *row_pointers = (png_bytep *)malloc(sizeof(png_bytep) * height);
    for (int y = 0; y < height; y++)
      row_pointers[y] = (png_byte *)malloc(png_get_rowbytes(png_ptr, info_ptr));

    for (int y = 0; y < height; y++) {
      png_byte *row = row_pointers[y];
      for (int x = 0; x < width; x++) {
        png_byte *ptr = &(row[x * 4]);
        ptr[0] = this->output_.weight_.get(x, y, 0);
        ptr[1] = this->output_.weight_.get(x, y, 1);
        ptr[2] = this->output_.weight_.get(x, y, 2);
        ptr[3] = 0xff;
      }
    }

    png_write_image(png_ptr, row_pointers);

    if (setjmp(png_jmpbuf(png_ptr)))
      return true;

    png_write_end(png_ptr, NULL);

    /* cleanup heap allocation */
    for (int y = 0; y < height; y++)
      free(row_pointers[y]);
    free(row_pointers);
    fclose(fp);
    return false;
  }

  void forward() override {}

  void backward() override {}
};

#endif // NOETHER_IMAGE_H
