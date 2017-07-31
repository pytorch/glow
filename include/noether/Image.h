#ifndef NOETHER_IMAGE_H
#define NOETHER_IMAGE_H

#include "noether/Node.h"
#include "noether/Tensor.h"

#include <cassert>
#include <cstddef>
#include <cstdint>
#include <string>

namespace noether {

class PNGNode final : public NodeBase {
public:
  PNGNode(Network *N) : NodeBase() {}

  virtual std::string getName() const override { return "PNGNode"; }

  bool writeImage(const char *filename);

  bool readImage(const char *filename);

  void init(Context *ctx) const override {}

  void forward(Context *ctx) const override {}

  void backward(Context *ctx) const override {}

  virtual void visit(NodeVisitor *visitor) override;
};

} // namespace

#endif // NOETHER_IMAGE_H
