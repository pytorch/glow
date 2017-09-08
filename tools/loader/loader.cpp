#include "glow/Importer/Caffe2.h"
#include "glow/Network/Image.h"
#include "glow/Network/Network.h"
#include "glow/Network/Nodes.h"
#include "glow/Network/Tensor.h"

using namespace glow;

/// Loads and normalizes a PNG into a tensor in the NCHW 3x224x224 format.
void loadImageAndPreprocess(const std::string &filename, Tensor *result) {
  Tensor localCopy;
  readPngImage(&localCopy, filename.c_str());
  auto catH = localCopy.getHandle<FloatTy>();

  auto dims = localCopy.dims();

  result->reset(ElemKind::FloatTy, {1, 3, dims[0], dims[1]});
  auto RH = result->getHandle<FloatTy>();

  for (unsigned z = 0; z < 3; z++) {
    for (unsigned y = 0; y < dims[1]; y++) {
      for (unsigned x = 0; x < dims[0]; x++) {
        // Convert to BGR, and subtract the mean, which we hardcode to 128.
        RH.at({0, 2 - z, x, y}) = (catH.at({x, y, z}) - 128);
      }
    }
  }
}

int main(int argc, char **argv) {
  if (argc != 4) {
    std::cerr << "Usage: " << argv[0]
              << "image.png network_structure.pb weights.pb\n";
    return -1;
  }

  Tensor data;
  Tensor expected_softmax(ElemKind::IndexTy, {1, 1});

  loadImageAndPreprocess(argv[1], &data);

  glow::Network N;
  caffe2ModelLoader LD(argv[2], argv[3], {"data", "softmax_expected"},
                       {&data, &expected_softmax}, N);

  auto *SM = LD.getNodeByName("prob");
  Variable *input = (Variable *)LD.getNodeByName("data");

  N.dumpGraph();

  auto *res = N.infer(SM, {input}, {&data});
  auto H = res->getHandle<FloatTy>();
  H.dump("res=", "\n");
  Tensor slice = H.extractSlice(0);
  auto SH = slice.getHandle<FloatTy>();

  std::cout << "\n";
  std::cout << "Result =" << SH.maxArg() << "\n";

  return 0;
}
