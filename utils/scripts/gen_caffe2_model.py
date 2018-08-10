## This is a helper script that generates Caffe2 models.
## The generated model will be used for Caffe2 importer unittest:    
## ./tests/unittests/caffe2ImporterTest.cpp 
## Run $>python gen_caffe2_model.py to get the model files.

from caffe2.proto import caffe2_pb2
from caffe2.python import utils
from google.protobuf import text_format
# Define a weights network
weights = caffe2_pb2.NetDef()
weights.name = "init"

op = caffe2_pb2.OperatorDef()
op.type = "GivenTensorFill"
op.output.extend(["conv_w"])
op.arg.extend([utils.MakeArgument("shape", [1, 1, 2, 2])])
op.arg.extend([utils.MakeArgument("values", [1.0 for i in range(4)])])
weights.op.extend([op])

op = caffe2_pb2.OperatorDef()
op.type = "GivenTensorFill"
op.output.extend(["conv_b"])
op.arg.extend([utils.MakeArgument("shape", [1])])
op.arg.extend([utils.MakeArgument("values", [2.0 for i in range(1)])])
weights.op.extend([op])
weights.external_output.extend(op.output)

# Define an inference net
net = caffe2_pb2.NetDef()
net.name = "predict"

op = caffe2_pb2.OperatorDef()
op.type = "Conv"
op.input.extend(["data"])
op.input.extend(["conv_w"])
op.input.extend(["conv_b"])
op.arg.add().CopyFrom(utils.MakeArgument("kernel", 2));
op.arg.add().CopyFrom(utils.MakeArgument("stride", 1));
op.arg.add().CopyFrom(utils.MakeArgument("group", 1));
op.arg.add().CopyFrom(utils.MakeArgument("pad", 1));
op.output.extend(["conv_out"])
net.op.extend([op])

net.external_output.extend(op.output)

# Generate model in text format.
with open('predict_net.pbtxt', 'w') as f:
  f.write(text_format.MessageToString(net));

with open('init_net.pbtxt', 'w') as f:
  f.write(text_format.MessageToString(weights))

# Generate model in binary format.
with open('predict_net.pb', 'wb') as f:
  f.write(net.SerializeToString());

with open('init_net.pb', 'wb') as f:
  f.write(net.SerializeToString())
