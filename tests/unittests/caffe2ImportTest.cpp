#include "glow/Graph/Graph.h"
#include "glow/IR/IR.h"
#include "glow/Importer/Caffe2.h"
#include "llvm/Support/CommandLine.h"

#include "gtest/gtest.h"

using namespace glow;

TEST(caffe2, import) {
  Graph G("SimpleImport");
  Module M(&G);

  std::string NetDescFilename("tests/models/caffe2ImportTest/predictNet.pb");
  std::string NetWeightFilename("tests/models/caffe2ImportTest/initNet.pb");

  SaveNode *SM;
  {
    caffe2ModelLoader LD(NetDescFilename, NetWeightFilename, {}, {}, G);
    SM = LD.getRoot();
  }

  // Optimize all of the dead code, removing one GenericNode
  auto numNodes = G.getNodes().size();
  ::glow::optimize(G, CompilationMode::Infer);
  EXPECT_EQ(G.getNodes().size(), numNodes - 1);
}
