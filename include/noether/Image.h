#ifndef NOETHER_IMAGE_H
#define NOETHER_IMAGE_H

#include "noether/Node.h"
#include "noether/Tensor.h"

#include <cassert>
#include <cstddef>
#include <cstdint>
#include <string>

namespace noether {

class PNGNode final : public TrainableNode {
public:
  PNGNode(Network *N) : TrainableNode(N) {  }

  virtual bool isTrainable() override { return false; }

  virtual std::string getName() const override { return "PNGNode"; }

  bool writeImage(const char *filename);

  bool readImage(const char *filename);

  void forward() override {}

  void backward() override {}

  virtual void visit(NodeVisitor *visitor) override;
};

} // namespace

#endif // NOETHER_IMAGE_H
