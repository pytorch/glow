#include "noether/Train.h"
#include "noether/Tensor.h"

using namespace noether;

void TrainableData::train(const TrainingConfig &config) {
  size_t batchSize = config.batchSize;
  float L1Decay = config.L1Decay;
  float L2Decay = config.L2Decay;
  float learningRate = config.learningRate;
  float momentum = config.momentum;

  // Do not change the weights of input layers that are marked as untrainable.
  if (!isTrainable_)
    return;

  /// If we are using the momentum technique then we need to allocate an array
  /// for the gradient sum.
  if (momentum > 0.0 && gsum_.size() == 0) {
    gsum_.reset(ElemKind::FloatTy, weight_.dims());
  }

  auto sz = weight_.size();
  auto W = weight_.getHandle<FloatTy>();
  auto G = gradient_.getHandle<FloatTy>();
  auto Gsum = gsum_.getHandle<FloatTy>();

  // For each weight/gradient pair:
  for (size_t x = 0; x < sz; x++) {
    // Do a simple SGD update:
    FloatTy L1Grad = L1Decay * (W.raw(x) > 0 ? 1 : -1);
    FloatTy L2Grad = L2Decay * (W.raw(x));
    FloatTy gij = (L2Grad + L1Grad + G.raw(x)) / batchSize;

    // Use the momentum to improve the gradient descent:
    // http://ufldl.stanford.edu/tutorial/supervised/OptimizationStochasticGradientDescent/
    if (momentum > 0.0) {
      // Momentum update:
      FloatTy dx = momentum * Gsum.raw(x) - learningRate * gij;
      // Save this value for the next iteration:
      Gsum.raw(x) = dx;
      // Apply the gradient.
      W.raw(x) += dx;
    } else {
      // Use regular SGD:
      W.raw(x) -= learningRate * gij;
    }
  }
}

void TrainableData::dump() {
  auto W = weight_.getHandle<FloatTy>();
  auto G = gradient_.getHandle<FloatTy>();
  auto Gsum = gsum_.getHandle<FloatTy>();

  W.dump("W");
  if (G.size())
    G.dump("G", "\n");
  if (Gsum.size())
    Gsum.dump("Gsum", "\n");
}

void TrainableData::verify() {
  if (gradient_.size()) {
    assert(gradient_.size() == weight_.size() &&
           "Gradient tensor does not match weight tensor");
  }
  if (gsum_.size()) {
    assert(gsum_.size() == weight_.size() &&
           "Gradient tensor does not match weight tensor");
  }
}
