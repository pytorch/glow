#include "glow/Base/Image.h"
#include "glow/Base/Tensor.h"
#include "glow/Base/IO.h"
#include "glow/Quantization/Base/Base.h"

using namespace glow;

int main(int argc, char **argv) {
  if (argc != 3) {
    fprintf(stderr, "Usage: png2bits INFILE OUTFILE\n");
    exit(1);
  }
  const char *filename = argv[1];
  const char *outfile = argv[2];
  Tensor png = readPngImageAndPreprocess(
    filename,
    ImageNormalizationMode::k0to1,
    ImageChannelOrder::BGR,
    ImageLayout::NCHW,
    false);
  TensorQuantizationParams TQP{1.0f / 127.0f, 0};
  Tensor qpng = quantization::quantizeTensor(png, TQP);
  writeToFile(qpng, outfile);
}
