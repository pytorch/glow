#ifndef GLOW_NETWORK_IMAGE_H
#define GLOW_NETWORK_IMAGE_H

#include "glow/Network/Node.h"
#include "glow/Network/Tensor.h"

#include <cassert>
#include <cstddef>
#include <cstdint>
#include <string>

namespace glow {

class PNGNode final : public NodeBase {
public:
  explicit PNGNode(Network *N) {}

  std::string getName() const override { return "PNGNode"; }

  std::string getDebugRepr(Context *ctx) const override;

  bool writeImage(const char *filename);

  bool readImage(const char *filename);

  void init(Context *ctx) const override {}

  void forward(Context *ctx, PassKind kind) const override {}

  void backward(Context *ctx) const override {}

  void visit(NodeBase *parent, NodeVisitor *visitor) override;
};

} // namespace glow

#endif // GLOW_NETWORK_IMAGE_H
