#include "noether/Train.h"
#include "noether/Network.h"
#include "noether/Tensor.h"

using namespace noether;

Trainer::~Trainer() {
  for (auto &p : gsum_) {
    delete p.second;
  }
}

void Trainer::train(TrainableData *trainable) {
  Tensor *weights = &trainable->weights_;
  Tensor *gradients = &trainable->gradients_;
  assert(weights->dims() == gradients->dims() && "Invalid tensor sizes");

  if (!trainable->isTrainable_)
    return;

  size_t batchSize = config.batchSize;
  float L1Decay = config.L1Decay;
  float L2Decay = config.L2Decay;
  float learningRate = config.learningRate;
  float momentum = config.momentum;

  auto sz = weights->size();
  auto W = weights->getHandle<FloatTy>();
  auto G = gradients->getHandle<FloatTy>();
  auto Gsum = Handle<FloatTy>::createInvalidHandle();

  /// If we are using the momentum technique then we need to allocate an array
  /// for the gradient sum.
  if (momentum > 0.0) {
    auto it = gsum_.find(gradients);

    if (it != gsum_.end()) {
      Gsum = it->second->getHandle<FloatTy>();
    } else {
      Tensor *gs = new Tensor();
      gs->reset(gradients);
      gsum_[gradients] = gs;
      Gsum = gs->getHandle<FloatTy>();
    }
  }

  // For each weight/gradient pair:
  for (size_t x = 0; x < sz; x++) {
    // Do a simple SGD update:
    FloatTy L1Grad = L1Decay * (W.raw(x) > 0 ? 1 : -1);
    FloatTy L2Grad = L2Decay * (W.raw(x));
    FloatTy gij = (L2Grad + L1Grad + G.raw(x)) / batchSize;

    // Use the momentum to improve the gradient descent:
    // http://ufldl.stanford.edu/tutorial/supervised/
    // OptimizationStochasticGradientDescent/
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
