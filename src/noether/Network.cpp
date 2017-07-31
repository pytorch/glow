#include "noether/Network.h"
#include "noether/Image.h"
#include "noether/Nodes.h"
#include "noether/Tensor.h"

#include <iostream>
#include <thread>
#include <unordered_set>

using namespace noether;

Context::~Context() {
  for (auto t : trainables_) {
    delete t.second;
  }
  for (auto t : tensors_) {
    delete t.second;
  }
}

Handle<FloatTy> Context::getWeightHandle(const TensorToken *tok) {
  return getTrainable(tok)->getWeightHandle();
}

Handle<FloatTy> Context::getGradHandle(const TensorToken *tok) {
  return getTrainable(tok)->getGradHandle();
}

void Context::allocateTrainable(const TensorToken *tok, bool trainable,
                                ArrayRef<size_t> dims) {
  assert(!trainables_.count(tok) && "Token already allocated");
  trainables_[tok] = new TrainableData(trainable, dims);
}

TrainableData *Context::getTrainable(const TensorToken *tok) {
  assert(trainables_.count(tok) && "The token was not allocated");
  return trainables_[tok];
}

void Context::allocateTensor(const TensorToken *tok, ElemKind kind,
                             ArrayRef<size_t> dims) {
  assert(!tensors_.count(tok) && "Token already allocated");
  tensors_[tok] = new Tensor(kind, dims);
}

Tensor *Context::getTensor(const TensorToken *tok) {
  assert(tensors_.count(tok) && "The token was not allocated");
  return tensors_[tok];
}

Network::Network() {
  for (unsigned i = 0, e = std::thread::hardware_concurrency(); i < e; i++) {
    state_.emplace_back(i);
  }
}

Network::~Network() {
  /// Delete the nodes of the network.
  for (auto *node : networkNodes_) {
    delete node;
  }
}

ConvNode *Network::createConvNode(NodeBase *input, size_t outDepth,
                                  size_t filterSize, size_t stride,
                                  size_t pad) {
  return addNode(new ConvNode(this, input, outDepth, filterSize, stride, pad));
}

MaxPoolNode *Network::createMaxPoolNode(NodeBase *input, size_t filterSize,
                                        size_t stride, size_t pad) {
  return addNode(new MaxPoolNode(this, input, filterSize, stride, pad));
}

FullyConnectedNode *Network::createFullyConnectedNode(NodeBase *input,
                                                      size_t outDepth) {
  return addNode(new FullyConnectedNode(this, input, outDepth));
}

RELUNode *Network::createRELUNode(NodeBase *input) {
  return addNode(new RELUNode(this, input));
}

SigmoidNode *Network::createSigmoidNode(NodeBase *input) {
  return addNode(new SigmoidNode(this, input));
}

SoftMaxNode *Network::createSoftMaxNode(NodeBase *input) {
  return addNode(new SoftMaxNode(this, input));
}

RegressionNode *Network::createRegressionNode(NodeBase *input) {
  return addNode(new RegressionNode(this, input));
}

MaxNode *Network::createMaxNode(NodeBase *input) {
  return addNode(new MaxNode(this, input));
}

ArrayNode *Network::createArrayNode(ArrayRef<size_t> dims) {
  return addNode(new ArrayNode(this, dims));
}

namespace {

struct BackwardPass : NodeVisitor {
  Context *ctx_;
  BackwardPass(Context *ctx) : ctx_(ctx) {}
  virtual void pre(NodeBase *N) override { N->backward(ctx_); }
};

struct ForwardPass : NodeVisitor {
  Context *ctx_;
  ForwardPass(Context *ctx) : ctx_(ctx) {}
  virtual void post(NodeBase *N) override { N->forward(ctx_); }
};

struct PrinterPass : NodeVisitor {
  virtual void post(NodeBase *N) override { std::cout << N->getName() << "->"; }
};

} // namespace

void Network::updateForwardBackward(Context *ctx, NodeBase *root, size_t start,
                                    size_t len, ArrayRef<NodeBase *> nodes,
                                    ArrayRef<Tensor *> inputs, bool isBatch) {
  for (size_t idx = 0; idx < len; idx++) {
    /// Update the inputs:
    for (int i = 0, e = nodes.size(); i < e; i++) {
      if (isBatch) {
        nodes[i]->updateInputs(ctx, inputs[i], start + idx);
      } else {
        nodes[i]->updateInput(ctx, inputs[i]);
      }
    }

    // Forward scan:
    ForwardPass FP(ctx);
    root->visit(&FP);

    // Backward scan in reverse order:
    BackwardPass BP(ctx);
    root->visit(&BP);
  }
}

void Network::learnGradient(Context *ctx) {
  for (auto &p : ctx->trainables_) {
    // Update the weights.
    trainer_.train(p.second);
    // Clear the gradients for the next round of training.
    p.second->clearGradient();
  }
}

static unsigned calculateNumThreads(unsigned numCores, unsigned batchSize) {
  unsigned best = 1;

  for (int i = 1; i < numCores; i++) {

    // The batch size must be a multiple of the number of threads or we'll skip
    /// some inputs.
    if (batchSize % i)
      continue;

    /// Each thread must handle at least 4 inputs.
    if ((batchSize / i) < 4)
      break;

    best = i;
  }
  return best;
}

/// Train the network starting with the node \p root. Perform \p iterations
/// iterations in the training loop. Update the nodes in \p nodes with the
/// values \p inputs.
void Network::train(NodeBase *root, size_t batches, ArrayRef<NodeBase *> nodes,
                    ArrayRef<Tensor *> inputs) {

  size_t batchSize = getConfig().batchSize;
  unsigned numThreads = calculateNumThreads(state_.size(), batchSize);
  unsigned sliceSize = batchSize / numThreads;

  std::vector<std::thread> threads;

  for (size_t i = 0; i < batches; i++) {
    /// Launch threads that update the different chunks in the batch:
    for (int t = 0; t < numThreads; t++) {
      /// Update the network inputs and perform the forward and backwards pass.
      threads.emplace_back([=] {
        updateForwardBackward(&state_[t], root, trainCounter_ + t * sliceSize,
                              sliceSize, nodes, inputs, true);
      });
    }

    /// Wait for the threads to finish.
    for (auto &t : threads) {
      t.join();
    }
    threads.clear();

    trainCounter_ += getConfig().batchSize;

    // The algorithm for merging the state from the different threads is
    /// described in the paper:
    // Alex Krizhevsky [2014]
    // One weird trick for parallelizing convolutional neural networks

    // Merge the gradients from all of the treads into the first thread.
    // For each buffer in the first tread:
    for (auto trainable : state_[0]) {
      // For each thread id.
      for (int tid = 1; tid < numThreads; tid++) {
        auto *T = state_[tid].getTrainable(trainable.first);
        trainable.second->mergeGradients(T);
        T->clearGradient();
      }
    }

    // Perform the delta updates on the first thread:
    learnGradient(&state_[0]);

    /// Send the calculated weights to all other threads:
    for (auto trainable : state_[0]) {
      for (int tid = 1; tid < numThreads; tid++) {
        auto *T = state_[tid].getTrainable(trainable.first);
        T->copyWeights(trainable.second);
      }
    }
  }
}

/// Perform a single training iteration for one input. Update the nodes in \p
/// nodes with the values \p inputs.
void Network::train(NodeBase *root, ArrayRef<NodeBase *> nodes,
                    ArrayRef<Tensor *> inputs) {
  assert(nodes.size() == inputs.size() && "Mismatched argument list");

  updateForwardBackward(&state_[0], root, trainCounter_, 1, nodes, inputs,
                        false);

  trainCounter_++;

  // Only update the gradient when we've reached the end of the batch.
  if (trainCounter_ % getConfig().batchSize)
    return;

  learnGradient(&state_[0]);
}

Tensor *Network::infer(NodeBase *root, ArrayRef<NodeBase *> nodes,
                       ArrayRef<Tensor *> inputs) {
  // Update all inputs.
  for (int i = 0, e = nodes.size(); i < e; i++) {
    nodes[i]->updateInput(&state_[0], inputs[i]);
  }

  // Forward scan.
  ForwardPass FP(&state_[0]);
  root->visit(&FP);

  return &root->getOutput(&state_[0])->weights_;
}

void Network::dump(NodeBase *root) {
  std::cout << "Network structure:";

  // Print all of the nodes in the network.
  PrinterPass FP;
  root->visit(&FP);

  std::cout << "\n";

  for (auto &ctx : state_) {
    std::cout << "Context:\n";
    for (auto &t : ctx) {
      t.second->getWeightHandle().dump("W:", "\n");
    }
  }

  std::cout << "\n";
}
