#ifndef NOETHER_TRAIN_H
#define NOETHER_TRAIN_H

#include "noether/ADT.h"

#include <cstddef>
#include <cstdint>
#include <unordered_map>


namespace noether {

class Tensor;

/// This is a list of parameters that the network trainers (such as sgd and
/// adam) use for training the network.
struct TrainingConfig {
  size_t batchSize{1};
  float L1Decay{0};
  float L2Decay{0};
  float learningRate{0.01};
  float momentum{0.0};
};

class Network;

class Trainer {
public:
  /// Holds the training configuration.
  TrainingConfig config{};
  
private:
  /// A temporary data structure for holding the attached gsum buffers.
  /// This data strucure owns the attached tensors, which are the values of
  /// the map, not the keys.
  std::unordered_map<Tensor*, Tensor*> gsum_;

public:

  Trainer() = default;

  ~Trainer();

  /// Perform a single iteration of the simple SGD algorithm for updating the
  /// weights of the program based on the gradients.
  void train(Tensor *weights, Tensor *gradients);
};

} // namespace

#endif // NOETHER_TRAIN_H
