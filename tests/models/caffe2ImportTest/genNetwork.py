from caffe2.proto import caffe2_pb2
from caffe2.python import utils

# Define a weights network
weights = caffe2_pb2.NetDef()
weights.name = "init"

op = caffe2_pb2.OperatorDef()
op.type = "fake_data_provider"
op.output.extend(["data"])
weights.op.extend([op])
weights.external_output.extend(op.output)

op = caffe2_pb2.OperatorDef()
op.type = "GivenTensorFill"
op.output.extend(["fc_w"])
op.arg.extend([utils.MakeArgument("shape", [1,4])])
op.arg.extend([utils.MakeArgument("values", [1.0 for i in range(4)])])
weights.op.extend([op])
weights.external_output.extend(op.output)

op = caffe2_pb2.OperatorDef()
op.type = "GivenTensorFill"
op.output.extend(["fc_b"])
op.arg.extend([utils.MakeArgument("shape", [1,4])])
op.arg.extend([utils.MakeArgument("values", [1.0 for i in range(4)])])
weights.op.extend([op])
weights.external_output.extend(op.output)

# Define an inference net
net = caffe2_pb2.NetDef()
net.name = "predict"

op = caffe2_pb2.OperatorDef()
op.type = "fake_operator"
op.input.extend(["data"])
op.output.extend(["fake_out"])
net.op.extend([op])

op = caffe2_pb2.OperatorDef()
op.type = "FC"
op.input.extend(["fake_out"])
op.input.extend(["fc_w"])
op.input.extend(["fc_b"])
op.output.extend(["fc_out"])
net.op.extend([op])

op = caffe2_pb2.OperatorDef()
op.type = "Relu"
op.input.extend(["fc_out"])
op.output.extend(["relu_out"])
net.op.extend([op])

# Relu out is what we want
net.external_output.extend(op.output)

# We want DCE to remove this one
op = caffe2_pb2.OperatorDef()
op.type = "useless_operator"
op.input.extend(["fake_out"])
op.output.extend(["useless_out"])
net.op.extend([op])

with open('predictNet.pb', 'wb') as f:
  f.write(net.SerializeToString())

with open('initNet.pb', 'wb') as f:
  f.write(weights.SerializeToString())
