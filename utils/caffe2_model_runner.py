# Copyright (c) Glow Contributors. See CONTRIBUTORS file.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from caffe2.proto import caffe2_pb2
from caffe2.python import workspace
from google.protobuf import text_format
import argparse


def fix_tensor_fills(init_net_file):
    init_net_pb = open(init_net_file, "rb").read()
    init_net = caffe2_pb2.NetDef()
    init_net.ParseFromString(init_net_pb)
    for op in init_net.op:
        if any("indices" in x for x in op.output):
            op.type = "GivenTensorInt64Fill"
        elif any("lengths" in x for x in op.output):
            op.type = "GivenTensorIntFill"
    open(
        init_net_file +
        "txt",
        "w").write(
        text_format.MessageToString(init_net))
    open(init_net_file, "wb").write(init_net.SerializeToString())


def read_init_net_pbtxt(init_net_file):
    init_net_txt = open(init_net_file, "r").read()
    init_net = caffe2_pb2.NetDef()
    text_format.Merge(init_net_txt, init_net)
    return init_net


def read_init_net(init_net_file):
    init_net_pb = open(init_net_file, "rb").read()
    init_net = caffe2_pb2.NetDef()
    init_net.ParseFromString(init_net_pb)
    return init_net


def read_predict_net(predict_net_file):
    predict_net_txt = open(predict_net_file, "r").read()
    predict_net = caffe2_pb2.NetDef()
    predict_net.name = "the_model"
    text_format.Merge(predict_net_txt, predict_net)
    return predict_net


def run(predict_net, init_net):
    workspace.ResetWorkspace()
    workspace.RunNetOnce(init_net)
    workspace.CreateNet(predict_net)
    workspace.RunNet(predict_net.name)
    out = workspace.FetchBlob(predict_net.external_output[0])
    print(out)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("predict_net", default="predict_net.pbtxt", nargs="?")
    parser.add_argument("init_net", default="init_net.pb", nargs="?")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    init_net = read_init_net(args.init_net)
    predict_net = read_predict_net(args.predict_net)
    run(predict_net, init_net)
