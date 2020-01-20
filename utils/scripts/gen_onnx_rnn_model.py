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

import onnx
from onnx import helper
from onnx import TensorProto
import numpy as np

import torch
import torch.nn
import torch.onnx

# RNN enums
RNN_DIR_FORWARD = 'forward'
RNN_DIR_REVERSE = 'reverse'
RNN_DIR_BIDIRECTIONAL = 'bidirectional'
RNN_DIRS = [RNN_DIR_FORWARD, RNN_DIR_REVERSE, RNN_DIR_BIDIRECTIONAL]


# ONNX utility
def make_init(name, type, tensor):
    return helper.make_tensor(name=name, data_type=type, dims=tensor.shape, vals=tensor.reshape(tensor.size).tolist())


# Function to generate RNN ONNX test model
def gen_rnn_onnx_test_model(model_path, seq_length, batch_size, hidden_size, input_size, direction, has_bias,
                            has_sequence_lens, has_initial_h):

    # Validate parameters
    assert direction in RNN_DIRS, 'ONNX RNN direction invalid!'
    assert not has_sequence_lens, 'ONNX RNN Variable sequence length not supported'

    # Get number of directions
    num_directions = 2 if (direction == RNN_DIR_BIDIRECTIONAL) else 1

    # Tensor sizes
    X_shape = [seq_length, batch_size, input_size]
    W_shape = [num_directions, 1 * hidden_size, input_size]
    R_shape = [num_directions, 1 * hidden_size, hidden_size]
    B_shape = [num_directions, 2 * hidden_size]
    sequence_lens_shape = [batch_size]
    initial_h_shape = [num_directions, batch_size, hidden_size]
    Y_shape = [seq_length, num_directions, batch_size, hidden_size]

    # Generate random inputs (weights are assumed concatenated in ONNX format: z,r,h)
    np.random.seed(1)
    X = np.random.randn(*X_shape)
    W = np.random.randn(*W_shape)
    R = np.random.randn(*R_shape)
    B = np.random.randn(*B_shape) if has_bias else np.zeros(B_shape)
    sequence_lens = np.random.randint(
        1, seq_length, batch_size) if has_sequence_lens else np.tile(seq_length, batch_size)
    initial_h = np.random.randn(
        *initial_h_shape) if has_initial_h else np.zeros(initial_h_shape)

    # Function to get all the weight components for the given direction
    def get_weights(dir_idx):
        Wi = np.reshape(W[dir_idx, 0 * hidden_size: 1 *
                          hidden_size, :], [hidden_size, input_size])
        Ri = np.reshape(R[dir_idx, 0 * hidden_size: 1 *
                          hidden_size, :], [hidden_size, hidden_size])
        bWi = np.reshape(B[dir_idx, 0 * hidden_size: 1 *
                           hidden_size], [hidden_size])
        bRi = np.reshape(B[dir_idx, 1 * hidden_size: 2 *
                           hidden_size], [hidden_size])
        return (Wi, Ri, bWi, bRi)

    # Function to get PyTorch weights (which are in the r,z,h order)
    def get_torch_weights(dir_idx):
        Wi, Ri, bWi, bRi = get_weights(dir_idx)
        W_torch = Wi
        R_torch = Ri
        bW_torch = bWi
        bR_torch = bRi
        return (W_torch, R_torch, bW_torch, bR_torch)

    # ----------------------------------------- COMPUTE pyTORCH REFERENCE ----------------------------------------------
    # Compute reference using Pytorch. Pytorch RNN has only forward/bidirectional so we will do the reverse RNN using
    # a Pytorch forward RNN.
    rnn = torch.nn.RNN(input_size=input_size,
                       hidden_size=hidden_size,
                       num_layers=1,
                       nonlinearity='tanh',
                       bias=True,
                       batch_first=False,
                       dropout=0,
                       bidirectional=(direction == RNN_DIR_BIDIRECTIONAL))

    # Get RNN state dictionary
    rnn_state_dict = rnn.state_dict()

    # Assign forward weights
    forwardEnabled = direction in [RNN_DIR_FORWARD, RNN_DIR_BIDIRECTIONAL]
    if forwardEnabled:
        forward_dir_idx = 0
        (W_torch, R_torch, bW_torch, bR_torch) = get_torch_weights(forward_dir_idx)
        rnn_state_dict['weight_ih_l0'] = torch.tensor(
            W_torch, dtype=torch.float32)
        rnn_state_dict['weight_hh_l0'] = torch.tensor(
            R_torch, dtype=torch.float32)
        rnn_state_dict['bias_ih_l0'] = torch.tensor(
            bW_torch, dtype=torch.float32)
        rnn_state_dict['bias_hh_l0'] = torch.tensor(
            bR_torch, dtype=torch.float32)

    # Assign reverse weights
    reverseEnabled = direction in [RNN_DIR_REVERSE, RNN_DIR_BIDIRECTIONAL]
    if reverseEnabled:
        if direction == RNN_DIR_REVERSE:
            reverse_dir_idx = 0
            (W_torch, R_torch, bW_torch, bR_torch) = get_torch_weights(reverse_dir_idx)
            rnn_state_dict['weight_ih_l0'] = torch.tensor(
                W_torch, dtype=torch.float32)
            rnn_state_dict['weight_hh_l0'] = torch.tensor(
                R_torch, dtype=torch.float32)
            rnn_state_dict['bias_ih_l0'] = torch.tensor(
                bW_torch, dtype=torch.float32)
            rnn_state_dict['bias_hh_l0'] = torch.tensor(
                bR_torch, dtype=torch.float32)
        else:
            reverse_dir_idx = 1
            (W_torch, R_torch, bW_torch, bR_torch) = get_torch_weights(reverse_dir_idx)
            rnn_state_dict['weight_ih_l0_reverse'] = torch.tensor(
                W_torch, dtype=torch.float32)
            rnn_state_dict['weight_hh_l0_reverse'] = torch.tensor(
                R_torch, dtype=torch.float32)
            rnn_state_dict['bias_ih_l0_reverse'] = torch.tensor(
                bW_torch, dtype=torch.float32)
            rnn_state_dict['bias_hh_l0_reverse'] = torch.tensor(
                bR_torch, dtype=torch.float32)

    # Set RNN state dictionary
    rnn.load_state_dict(rnn_state_dict, strict=True)

    # Perform inference
    X_torch = torch.tensor(X, dtype=torch.float32)
    initial_h_torch = torch.tensor(initial_h, dtype=torch.float32)
    if direction == RNN_DIR_REVERSE:
        Y, next_h = rnn(X_torch.flip([0]), initial_h_torch)
        Y = Y.flip([0])
    else:
        Y, next_h = rnn(X_torch, initial_h_torch)

    # Reshape output to ONNX format [seq_length, num_directions, batch_size, hidden_size]
    Y_ref = Y.detach().numpy()
    Y_ref = np.reshape(
        Y_ref, [seq_length, batch_size, num_directions, hidden_size])
    Y_ref = np.transpose(Y_ref, [0, 2, 1, 3])

    # Reshape states to ONNX format
    Y_h_ref = next_h.detach().numpy()

    # --------------------------------------- COMPUTE PYTHON-NUMPY REFERENCE -------------------------------------------
    # Create X slices
    Xslices = list()
    for t in range(seq_length):
        Xslices.append(np.reshape(X[t, :, :], [batch_size, input_size]))

    # Function to compute one RNN cell
    def compute_rnn(forward):
        dir_idx = 0 if forward else (0 if direction == RNN_DIR_REVERSE else 1)
        Wi, Ri, bWi, bRi = get_weights(dir_idx)

        def f(x): return np.tanh(x)

        def mm(x, w): return np.matmul(x, w.transpose())
        Ht = np.reshape(initial_h[dir_idx, :, :], [batch_size, hidden_size])

        Yslices = list()
        for t in range(seq_length):
            xt = Xslices[t] if forward else Xslices[seq_length - 1 - t]
            Ht = f(mm(xt, Wi) + bWi + mm(Ht, Ri) + bRi)
            Yslices.append(Ht)
        return Yslices, Ht

    Yslices = list()
    Hslices = list()

    # Compute forward RNN
    forwardYslices = list()
    if forwardEnabled:
        Yt, Ht = compute_rnn(True)
        forwardYslices += Yt
        Hslices.append(Ht)

    # Compute reverse RNN
    reverseYslices = list()
    if reverseEnabled:
        Yt, Ht = compute_rnn(False)
        reverseYslices += Yt
        Hslices.append(Ht)

    # Concatenate slices
    for t in range(seq_length):
        if forwardEnabled:
            Yslices.append(forwardYslices[t])
        if reverseEnabled:
            Yslices.append(reverseYslices[seq_length - 1 - t])
    Y_ref_np = np.concatenate(Yslices, 0).reshape(
        [seq_length, num_directions, batch_size, hidden_size])
    Y_h_ref_np = np.concatenate(Hslices, 0).reshape(
        [num_directions, batch_size, hidden_size])

    # Compare Numpy with Torch implementation.
    assert np.max(np.abs(Y_ref - Y_ref_np)
                  ) < 1e-6, "Mismatch between Pytorch and Numpy RNN implementation"
    assert np.max(np.abs(Y_h_ref - Y_h_ref_np)
                  ) < 1e-6, "Mismatch between Pytorch and Numpy RNN implementation"

    # ---------------------------------------------- NODE DEFINITION  --------------------------------------------------
    # Node inputs
    node_inputs = ['X',
                   'W',
                   'R',
                   'B' if has_bias else '',
                   '',
                   'initial_h' if has_initial_h else '']

    # Node outputs
    node_outputs = ['Y', 'Y_h']

    # RNN node definition
    rnn_node_def = onnx.helper.make_node(
        'RNN',
        name='rnn',
        inputs=node_inputs,
        outputs=node_outputs,
        hidden_size=hidden_size,
        direction=direction
    )

    # Error node definition
    err_node_def = onnx.helper.make_node(
        'Sub',
        name='error',
        inputs=['Y', 'Y_ref'],
        outputs=['Y_err']
    )

    # --------------------------------------------- GRAPH DEFINITION  --------------------------------------------------
    graph_input = list()
    graph_init = list()
    graph_output = list()

    # RNN inputs
    graph_input.append(helper.make_tensor_value_info(
        'X', TensorProto.FLOAT, X_shape))
    graph_input.append(helper.make_tensor_value_info(
        'W', TensorProto.FLOAT, W_shape))
    graph_input.append(helper.make_tensor_value_info(
        'R', TensorProto.FLOAT, R_shape))
    if has_bias:
        graph_input.append(helper.make_tensor_value_info(
            'B', TensorProto.FLOAT, B_shape))
    if has_sequence_lens:
        graph_input.append(helper.make_tensor_value_info(
            'sequence_lens', TensorProto.INT32, sequence_lens_shape))
    if has_initial_h:
        graph_input.append(helper.make_tensor_value_info(
            'initial_h', TensorProto.FLOAT, initial_h_shape))

    # Reference input
    graph_input.append(helper.make_tensor_value_info(
        'Y_ref', TensorProto.FLOAT, Y_shape))

    # RNN initializers
    graph_init.append(make_init('X', TensorProto.FLOAT, X))
    graph_init.append(make_init('W', TensorProto.FLOAT, W))
    graph_init.append(make_init('R', TensorProto.FLOAT, R))
    if has_bias:
        graph_init.append(make_init('B', TensorProto.FLOAT, B))
    if has_sequence_lens:
        graph_init.append(
            make_init('sequence_lens', TensorProto.INT32, sequence_lens))
    if has_initial_h:
        graph_init.append(make_init('initial_h', TensorProto.FLOAT, initial_h))

    # Reference initializer
    graph_init.append(make_init('Y_ref', TensorProto.FLOAT, Y_ref))

    # Graph outputs
    graph_output.append(helper.make_tensor_value_info(
        'Y_err', TensorProto.FLOAT, Y_shape))

    # Define graph (GraphProto)
    graph_name = 'rnn_test'
    graph_def = helper.make_graph(
        [rnn_node_def, err_node_def], graph_name, inputs=graph_input, outputs=graph_output)

    # Set initializers
    graph_def.initializer.extend(graph_init)

    # --------------------------------------------- MODEL DEFINITION  --------------------------------------------------
    # Define model (ModelProto)
    model_def = helper.make_model(graph_def, producer_name='onnx-rnn')

    # Check model
    onnx.checker.check_model(model_def)

    # Print model
    with open(model_path, 'w') as f:
        f.write(str(model_def))


# Forward RNN
gen_rnn_onnx_test_model(model_path='rnnForward.onnxtxt',
                        seq_length=2,
                        batch_size=5,
                        hidden_size=4,
                        input_size=3,
                        direction='forward',
                        has_bias=True,
                        has_sequence_lens=False,
                        has_initial_h=True)

# Reverse RNN
gen_rnn_onnx_test_model(model_path='rnnReverse.onnxtxt',
                        seq_length=2,
                        batch_size=5,
                        hidden_size=4,
                        input_size=3,
                        direction='reverse',
                        has_bias=True,
                        has_sequence_lens=False,
                        has_initial_h=True)

# Bidirectional RNN
gen_rnn_onnx_test_model(model_path='rnnBidirectional.onnxtxt',
                        seq_length=2,
                        batch_size=5,
                        hidden_size=4,
                        input_size=3,
                        direction='bidirectional',
                        has_bias=True,
                        has_sequence_lens=False,
                        has_initial_h=True)

# Forward no bias RNN
gen_rnn_onnx_test_model(model_path='rnnForwardNoBias.onnxtxt',
                        seq_length=1,
                        batch_size=5,
                        hidden_size=4,
                        input_size=3,
                        direction='forward',
                        has_bias=False,
                        has_sequence_lens=False,
                        has_initial_h=True)

# Forward no state RNN
gen_rnn_onnx_test_model(model_path='rnnForwardNoState.onnxtxt',
                        seq_length=1,
                        batch_size=5,
                        hidden_size=4,
                        input_size=3,
                        direction='forward',
                        has_bias=True,
                        has_sequence_lens=False,
                        has_initial_h=False)
