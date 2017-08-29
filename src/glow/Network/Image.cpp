#include "glow/Network/Image.h"
#include "glow/Network/Nodes.h"
#include "glow/Network/Tensor.h"
#include "glow/Support/Support.h"

using namespace glow;

std::string PNGNode::getDebugRepr(Context *ctx) const {
  DescriptionBuilder db(getName());
  db.addDim("output", getOutputWeight(ctx)->dims());
  return db;
}

#if (GLOW_PNG_FOUND)
#include <png.h>

/// Reads a png image. \returns True if an error occurred.
bool PNGNode::readImage(const char *filename) {
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
  (void)bit_depth;
  assert(bit_depth == 8 && "Invalid image");
  assert(color_type == PNG_COLOR_TYPE_RGB_ALPHA ||
         color_type == PNG_COLOR_TYPE_RGB && "Invalid image");
  bool hasAlpha = (color_type == PNG_COLOR_TYPE_RGB_ALPHA);

  int number_of_passes = png_set_interlace_handling(png_ptr);
  (void)number_of_passes;
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

  output_.weight_.reset(width, height, 3);

  for (int y = 0; y < height; y++) {
    png_byte *row = row_pointers[y];
    for (int x = 0; x < width; x++) {
      png_byte *ptr = &(row[x * (hasAlpha ? 4 : 3)]);

      output_.weight_.at(x, y, 0) = ptr[0];
      output_.weight_.at(x, y, 1) = ptr[1];
      output_.weight_.at(x, y, 2) = ptr[2];
    }
  }

  for (int y = 0; y < height; y++)
    free(row_pointers[y]);
  free(row_pointers);

  return false;
}

bool PNGNode::writeImage(const char *filename) {
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

  auto odim = output_.dims();
  assert(odim[2] < 4 && "Invalid buffer to save");

  int width = odim[0];
  int height = odim[1];
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
      ptr[0] = output_.weight_.at(x, y, 0);
      ptr[1] = output_.weight_.at(x, y, 1);
      ptr[2] = output_.weight_.at(x, y, 2);
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

void PNGNode::visit(NodeVisitor *visitor) {
  if (!visitor->shouldVisit(this))
    return;
  visitor->pre(this);
  visitor->post(this);
}

#else
bool PNGNode::writeImage(const char *filename) {
  assert(false && "Not configured with libpng");
}

bool PNGNode::readImage(const char *filename) {
  assert(false && "Not configured with libpng");
}
#endif
