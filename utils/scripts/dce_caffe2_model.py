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
from google.protobuf import text_format
import argparse


def read_model_from_file(path):
    m = caffe2_pb2.NetDef()
    with open(path, "rb") as f:
        if ".pbtxt" in path:
            text_format.Merge(f.read(), m)
        else:
            m.ParseFromString(f.read())
    return m


def write_model_to_file(path, m):
    with open(path, "wb") as f:
        if ".pbtxt" in path:
            f.write(text_format.MessageToString(m))
        else:
            f.write(m.SerializeToString())


# Perform dead code elimination on predict_net removing any nodes that aren't
# used for producing values in predict_net.external_output. Remove any nodes in
# init_net that produce values that are no longer needed by predict_net.


def dce(init_net, predict_net):
    num_predict_net_ops_original = len(predict_net.op)
    num_predict_net_inputs_original = len(predict_net.external_input)

    # Find the set of tensors used in the computation of the outputs.
    live_predict_net_op_outputs = set(predict_net.external_output)
    prev_num_live_predict_net_op_outputs = len(live_predict_net_op_outputs)
    while True:
        for op in predict_net.op:
            for output_tensor in op.output:
                if output_tensor in live_predict_net_op_outputs:
                    for input_tensor in op.input:
                        live_predict_net_op_outputs.add(input_tensor)
        num_live_predict_net_op_outputs = len(live_predict_net_op_outputs)
        if num_live_predict_net_op_outputs == prev_num_live_predict_net_op_outputs:
            break
        prev_num_live_predict_net_op_outputs = num_live_predict_net_op_outputs

    # Find the ops that are required to compute the tensors used during
    # computation of the outputs.
    live_predict_net_ops = []
    for op in predict_net.op:
        for output_tensor in op.output:
            if output_tensor in live_predict_net_op_outputs:
                live_predict_net_ops.append(op)

    # Delete all unused ops in predict_net.
    num_predict_net_ops_eliminated = len(
        predict_net.op) - len(live_predict_net_ops)
    del predict_net.op[:]
    predict_net.op.extend(live_predict_net_ops)

    # Find the set of all used inputs tensors in predict_net.
    live_predict_net_op_inputs = set()
    for op in predict_net.op:
        for input_tensor in op.input:
            live_predict_net_op_inputs.add(input_tensor)

    # Find the set of used external_inputs.
    live_predict_net_external_inputs = set()
    for external_input in predict_net.external_input:
        if external_input in live_predict_net_op_inputs:
            live_predict_net_external_inputs.add(external_input)

    # Delete unused external_inputs in predict_net.
    num_predict_net_inputs_eliminated = len(predict_net.external_input) - len(
        live_predict_net_external_inputs
    )
    del predict_net.external_input[:]
    predict_net.external_input.extend(live_predict_net_external_inputs)

    print(
        "predict_net ops eliminated: {}/{}".format(
            num_predict_net_ops_eliminated, num_predict_net_ops_original
        )
    )
    print(
        "predict_net external_inputs eliminated: {}/{}".format(
            num_predict_net_inputs_eliminated, num_predict_net_inputs_original
        )
    )

    # Everything below pertains to removing unused outputs in the init_net,
    # if no init net was provided then stop here.
    if init_net is None:
        return

    num_init_net_ops_original = len(init_net.op)

    # Find the set of init_net ops with outputs needed by the init_net
    live_init_net_ops = []
    for op in init_net.op:
        for output_tensor in op.output:
            if output_tensor in live_predict_net_external_inputs:
                live_init_net_ops.append(op)

    # Eliminate dead init_net ops
    num_init_net_ops_eliminated = len(init_net.op) - len(live_init_net_ops)
    del init_net.op[:]
    init_net.op.extend(live_init_net_ops)

    # Update init_net external_outputs
    live_init_net_op_outputs = set()
    for op in init_net.op:
        for output_tensor in op.output:
            live_init_net_op_outputs.add(output_tensor)

    live_init_net_external_outputs = set()
    for output_tensor in init_net.external_output:
        if output_tensor in live_init_net_op_outputs:
            live_init_net_external_outputs.add(output_tensor)

    del init_net.external_output[:]
    init_net.external_output.extend(live_init_net_external_outputs)

    print(
        "init_net ops eliminated: {}/{}".format(
            num_init_net_ops_eliminated, num_init_net_ops_original
        )
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Caffe2 model dead code elimination")
    parser.add_argument("--input_init_net_path", type=str)
    parser.add_argument("--input_predict_net_path", type=str, required=True)
    parser.add_argument("--output_init_net_path", type=str)
    parser.add_argument("--output_predict_net_path", type=str, required=True)

    args = parser.parse_args()

    predict_net = read_model_from_file(args.input_predict_net_path)

    init_net = None
    if args.input_init_net_path is not None:
        init_net = read_model_from_file(args.input_init_net_path)

    dce(init_net, predict_net)

    write_model_to_file(args.output_predict_net_path, predict_net)

    if args.output_init_net_path is not None:
        write_model_to_file(args.output_init_net_path, init_net)
