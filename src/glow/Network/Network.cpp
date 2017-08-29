#include "glow/Network/Network.h"
#include "glow/Network/Image.h"
#include "glow/Network/Nodes.h"
#include "glow/Network/Tensor.h"
#include "glow/Support/Support.h"

#include <fstream>
#include <iostream>
#include <thread>
#include <unordered_set>

using namespace glow;

Context::~Context() {
  for (auto t : tensors_) {
    delete t.second;
  }
}

Tensor *Context::allocateTensor(const TensorToken *tok, ElemKind kind,
                                ArrayRef<size_t> dims, ShareKind shared) {
  /// If we are asked to allocate a shared tensor then make sure to allocate it
  /// in the main context.
  if (shared == ShareKind::kSharedTensor && primeCtx_) {
    return nullptr;
  }

  assert(!hasTensor(tok) && "Token already allocated");
  Tensor *T = new Tensor(kind, dims);
  tensors_[tok] = T;
  return T;
}

Tensor *Context::getTensor(const TensorToken *tok) {
  // Look for the tensor in the local storage.
  auto it = tensors_.find(tok);
  if (it != tensors_.end()) {
    return it->second;
  }

  // If we could not find the tensor, search the prime context for shared
  // tensors.
  assert(primeCtx_ && "Could not find the tensor in the prime context!");
  return primeCtx_->getTensor(tok);
}

bool Context::hasTensor(const TensorToken *tok) { return tensors_.count(tok); }

Handle<FloatTy> Context::getHandle(const TensorToken *tok) {
  return getTensor(tok)->getHandle<FloatTy>();
}

Network::Network() {
  state_.push_back(new Context(nullptr));

  for (unsigned i = 1, e = std::thread::hardware_concurrency(); i < e; i++) {
    state_.push_back(new Context(state_[0]));
  }
}

Network::~Network() {
  /// Delete the nodes of the network.
  for (auto *node : networkNodes_) {
    delete node;
  }

  /// Delete the context.
  for (auto *ctx : state_) {
    delete ctx;
  }
}

void Network::updateTensor(const TensorToken *tok, Tensor *t) {
  getMainContext()->getTensor(tok)->copyFrom(t);
}

ConvNode *Network::createConvNode(NodeBase *input, size_t outDepth,
                                  size_t filterSize, size_t stride,
                                  size_t pad) {
  return addNode(new ConvNode(this, input, outDepth, filterSize, stride, pad));
}

ConcatNode *Network::createConcatNode(ArrayRef<NodeBase *> inputs,
                                      unsigned dimension) {
  return addNode(new ConcatNode(this, inputs, dimension));
}

MaxPoolNode *Network::createMaxPoolNode(NodeBase *input,
                                        MaxPoolNode::OpKind kind,
                                        size_t filterSize, size_t stride,
                                        size_t pad) {
  return addNode(new MaxPoolNode(this, input, kind, filterSize, stride, pad));
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

SoftMaxNode *Network::createSoftMaxNode(NodeBase *input, NodeBase *selected) {
  return addNode(new SoftMaxNode(this, input, selected));
}

RegressionNode *Network::createRegressionNode(NodeBase *input,
                                              NodeBase *expected) {
  return addNode(new RegressionNode(this, input, expected));
}

MaxNode *Network::createMaxNode(NodeBase *input) {
  return addNode(new MaxNode(this, input));
}

Variable *Network::createVariable(ArrayRef<size_t> dims, ElemKind elemTy) {
  return addNode(new Variable(this, dims, elemTy));
}

ReshapeNode *Network::createReshapeNode(NodeBase *input,
                                        ArrayRef<size_t> shape) {
  return addNode(new ReshapeNode(this, input, shape));
}

BatchNormalizationNode *
Network::createBatchNormalizationNode(NodeBase *input, size_t channelIdx,
                                      FloatTy epsilon, FloatTy momentum) {
  return addNode(
      new BatchNormalizationNode(this, input, channelIdx, epsilon, momentum));
}

ArithmeticNode *Network::createArithmeticNode(NodeBase *LHS, NodeBase *RHS,
                                              ArithmeticNode::OpKind op) {
  return addNode(new ArithmeticNode(this, LHS, RHS, op));
}

namespace {

/// A visitor class that collects a reverse post order of the nodes in the
/// graph.
class TopologicalSortPass : public NodeVisitor {
  std::unordered_set<NodeBase *> visited;
  std::vector<NodeBase *> order;

public:
  // Don't revisit visited nodes.
  bool shouldVisit(NodeBase *parent, NodeBase *N) override {
    return !visited.count(N);
  }

  TopologicalSortPass() = default;

  void post(NodeBase *parent, NodeBase *N) override {
    if (!visited.insert(N).second)
      return;

    order.push_back(N);
  }

  ArrayRef<NodeBase *> getOrder() { return order; }
};

struct PrinterPass : NodeVisitor {
  void post(NodeBase *parent, NodeBase *N) override {
    std::cout << N->getName() << "->";
  }
};

} // namespace

void Network::updateForwardBackward(Context *ctx, NodeBase *root,
                                    size_t sampleIdx, ArrayRef<Variable *> vars,
                                    ArrayRef<Tensor *> inputs) {
  TopologicalSortPass TPS;
  root->visit(nullptr, &TPS);
  auto order = TPS.getOrder();

  /// Update the inputs:
  for (int i = 0, e = vars.size(); i < e; i++) {
    vars[i]->updateInputs(ctx, inputs[i], sampleIdx);
  }

  for (auto it = order.begin(), e = order.end(); it != e; it++) {
    // Prepare for the next backprop iteration by zeroing the gradient
    // tensors. Notice that this only zeros the temporary grad tensors that
    // match the output tensors but not the gradient tensors that are
    // paired with filters. These are cleared during the learning process
    // at the end of the batch.
    (*it)->clearOutputGrad(ctx);

    // Perform the learning phase.
    (*it)->forward(ctx, NodeBase::PassKind::kTraining);
  }

  for (auto it = order.rbegin(), e = order.rend(); it != e; it++) {
    (*it)->backward(ctx);
  }
}

void Network::learnGradient(Context *ctx, size_t batchSize) {
  for (auto p : ctx->getTensorPairs()) {
    Tensor *W = ctx->getTensor(p.first);
    Tensor *G = ctx->getTensor(p.second);
    trainer_.train(W, G, batchSize);
  }
}

static unsigned calculateNumThreads(unsigned maxNumThreads, unsigned numCores,
                                    unsigned numPackets) {
  unsigned best = 1;
  assert(maxNumThreads && numCores && numPackets &&
         "Invalid work size or thread count");
  unsigned maxThreads = std::min<unsigned>(numCores, maxNumThreads);

  for (int i = 1; i < maxThreads; i++) {
    // The number of packets must be a multiple of the number of threads or
    // we'll skip some packets.
    if (numPackets % i) {
      continue;
    }

    /// Each thread must handle at least 2 packets.
    if ((numPackets / i) < 2) {
      break;
    }

    best = i;
  }
  return best;
}

/// Train the network starting with the node \p root. Update the vars in \p vars
/// with the values \p inputs. Train the network by processing \p numBatches
/// batches.
void Network::train(NodeBase *root, size_t numBatches,
                    ArrayRef<Variable *> vars, ArrayRef<Tensor *> inputs) {
  assert(inputs.size() && "No inputs");
  assert(inputs.size() == vars.size() &&
         "The number of inputs does not match the number of variables");

  // This is the size of one batch (the number of samples in the batch).
  size_t batchSize = vars[0]->dims(getMainContext())[0];

  // Figure out how many threads to use when training the network.
  unsigned numThreads =
      calculateNumThreads(getConfig().maxNumThreads, state_.size(), numBatches);

  std::vector<std::thread> threads;

  for (size_t i = 0; i < numBatches / numThreads; i++) {
    // Launch threads that update the different chunks in the batch:
    for (int t = 0; t < numThreads; t++) {
      // Update the network inputs and perform the forward and backwards pass.
      threads.emplace_back([=] {
        updateForwardBackward(state_[t], root, trainCounter_ + t * batchSize,
                              vars, inputs);
      });
    }

    /// Wait for the threads to finish.
    for (auto &t : threads) {
      t.join();
    }
    threads.clear();

    trainCounter_ += numThreads * batchSize;

    // The algorithm for merging the state from the different threads is
    /// described in the paper: Alex Krizhevsky [2014]
    // "One weird trick for parallelizing convolutional neural networks"
    for (int tid = 0; tid < numThreads; tid++) {
      learnGradient(state_[tid], batchSize * numThreads);
    }
  }
}

Tensor *Network::infer(NodeBase *root, ArrayRef<Variable *> vars,
                       ArrayRef<Tensor *> inputs) {
  TopologicalSortPass TPS;
  root->visit(nullptr, &TPS);
  auto order = TPS.getOrder();

  // Update all inputs.
  for (int i = 0, e = vars.size(); i < e; i++) {
    vars[i]->updateInputs(state_[0], inputs[i], 0);
  }

  // Forward scan.
  for (auto it = order.begin(), e = order.end(); it != e; it++) {
    (*it)->forward(state_[0], NodeBase::PassKind::kInference);
  }

  return root->getOutputWeight(state_[0]);
}

void Network::dump() {
  std::cout << "Network structure:";

  for (auto &N : networkNodes_) {
    std::cout << N->getName() << "\n";
  }

  std::cout << "\n";

  for (auto &ctx : state_) {
    std::cout << "Context:\n";
    for (auto &t : *ctx) {
      t.second->getHandle<FloatTy>().dump("W:", "\n");
    }
  }

  std::cout << "\n";
}

struct DottyPrinterPass : NodeVisitor {
  Network *net_;
  using edgeTy = std::pair<NodeBase *, NodeBase *>;
  std::vector<edgeTy> nodeEdges;
  DottyPrinterPass(Network *net) : net_(net) {}

public:
  // Don't revisit visited nodes.
  bool shouldVisit(NodeBase *parent, NodeBase *N) override {
    edgeTy e = {parent, N};
    return std::find(nodeEdges.begin(), nodeEdges.end(), e) == nodeEdges.end();
  }

  DottyPrinterPass() = default;

  void pre(NodeBase *parent, NodeBase *N) override {
    nodeEdges.push_back({parent, N});
  }

  std::string nodeDescr(NodeBase *N) {
    if (!N)
      return "";

    Context *ctx = net_->getMainContext();
    // Print a node descriptor that looks like this:
    // Format: "node12" [ label = "0xf7fc43e01" shape = "record" ];
    std::string sb;
    sb += quote(pointerToString(N)) + "[\n";
    std::string repr = escapeDottyString(N->getDebugRepr(ctx));
    sb += "\tlabel = " + repr + "\n";
    sb += "\tshape = \"record\"\n";
    sb += "];\n\n";
    return sb;
  }

  std::string quote(std::string in) { return '"' + in + '"'; }
  std::string getDottyString() {
    std::string sb;

    sb += "digraph finite_state_machine {\n\trankdir=TD;\n";

    // Assign a unique name to each one of the nodes:
    for (auto &e : nodeEdges) {
      if (e.first) {
        sb += quote(pointerToString(e.second)) + " -> " +
              quote(pointerToString(e.first)) + ";\n";
      }
    }

    // Assign a unique name to each one of the nodes:
    for (auto &e : nodeEdges) {
      sb += nodeDescr(e.first);
      sb += nodeDescr(e.second);
    }

    sb += "}";
    return sb;
  }
};

void Network::dumpGraph() {
  DottyPrinterPass DP(this);

  for (auto &N : networkNodes_) {
    N->visit(nullptr, &DP);
  }

  std::string filename = "dotty_network_dump_" + pointerToString(this) + ".dot";
  std::cout << "Writing dotty graph to: " << filename << '\n';

  std::string rep = DP.getDottyString();

  std::ofstream myfile;
  myfile.open(filename);
  myfile << rep;
  myfile.close();
}
