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

# GRU enums
GRU_DIR_FORWARD = 'forward'
GRU_DIR_REVERSE = 'reverse'
GRU_DIR_BIDIRECTIONAL = 'bidirectional'
GRU_DIRS = [GRU_DIR_FORWARD, GRU_DIR_REVERSE, GRU_DIR_BIDIRECTIONAL]


# ONNX utility
def make_init(name, type, tensor):
    return helper.make_tensor(name=name, data_type=type, dims=tensor.shape, vals=tensor.reshape(tensor.size).tolist())


# Function to generate GRU ONNX test model
def gen_gru_onnx_test_model(model_path, seq_length, batch_size, hidden_size, input_size, direction, has_bias,
                            has_sequence_lens, has_initial_h, linear_before_reset=False):

    # Validate parameters
    assert direction in GRU_DIRS, 'ONNX GRU direction invalid!'
    assert not has_sequence_lens, 'ONNX GRU Variable sequence length not supported'

    # Get number of directions
    num_directions = 2 if (direction == GRU_DIR_BIDIRECTIONAL) else 1

    # Tensor sizes
    X_shape = [seq_length, batch_size, input_size]
    W_shape = [num_directions, 3 * hidden_size, input_size]
    R_shape = [num_directions, 3 * hidden_size, hidden_size]
    B_shape = [num_directions, 6 * hidden_size]
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
        Wz = np.reshape(W[dir_idx, 0 * hidden_size: 1 *
                          hidden_size, :], [hidden_size, input_size])
        Wr = np.reshape(W[dir_idx, 1 * hidden_size: 2 *
                          hidden_size, :], [hidden_size, input_size])
        Wh = np.reshape(W[dir_idx, 2 * hidden_size: 3 *
                          hidden_size, :], [hidden_size, input_size])
        Rz = np.reshape(R[dir_idx, 0 * hidden_size: 1 *
                          hidden_size, :], [hidden_size, hidden_size])
        Rr = np.reshape(R[dir_idx, 1 * hidden_size: 2 *
                          hidden_size, :], [hidden_size, hidden_size])
        Rh = np.reshape(R[dir_idx, 2 * hidden_size: 3 *
                          hidden_size, :], [hidden_size, hidden_size])
        bWz = np.reshape(B[dir_idx, 0 * hidden_size: 1 *
                           hidden_size], [hidden_size])
        bWr = np.reshape(B[dir_idx, 1 * hidden_size: 2 *
                           hidden_size], [hidden_size])
        bWh = np.reshape(B[dir_idx, 2 * hidden_size: 3 *
                           hidden_size], [hidden_size])
        bRz = np.reshape(B[dir_idx, 3 * hidden_size: 4 *
                           hidden_size], [hidden_size])
        bRr = np.reshape(B[dir_idx, 4 * hidden_size: 5 *
                           hidden_size], [hidden_size])
        bRh = np.reshape(B[dir_idx, 5 * hidden_size: 6 *
                           hidden_size], [hidden_size])
        return Wz, Wr, Wh, Rz, Rr, Rh, bWz, bWr, bWh, bRz, bRr, bRh

    # Function to get PyTorch weights (which are in the r,z,h order)
    def get_torch_weights(dir_idx):
        Wz, Wr, Wh, Rz, Rr, Rh, bWz, bWr, bWh, bRz, bRr, bRh = get_weights(
            dir_idx)
        W_torch = np.concatenate((Wr, Wz, Wh), 0)
        R_torch = np.concatenate((Rr, Rz, Rh), 0)
        bW_torch = np.concatenate((bWr, bWz, bWh), 0)
        bR_torch = np.concatenate((bRr, bRz, bRh), 0)
        return (W_torch, R_torch, bW_torch, bR_torch)

    # ----------------------------------------- COMPUTE pyTORCH REFERENCE ----------------------------------------------
    # Compute reference using Pytorch. Pytorch GRU has only forward/bidirectional so we will do the reverse GRU using
    # a Pytorch forward GRU.
    gru = torch.nn.GRU(input_size=input_size,
                       hidden_size=hidden_size,
                       num_layers=1,
                       bias=True,
                       batch_first=False,
                       dropout=0,
                       bidirectional=(direction == GRU_DIR_BIDIRECTIONAL))

    # Get GRU state dictionary
    gru_state_dict = gru.state_dict()

    # Assign forward weights
    forwardEnabled = direction in [GRU_DIR_FORWARD, GRU_DIR_BIDIRECTIONAL]
    if forwardEnabled:
        forward_dir_idx = 0
        (W_torch, R_torch, bW_torch, bR_torch) = get_torch_weights(forward_dir_idx)
        gru_state_dict['weight_ih_l0'] = torch.tensor(
            W_torch, dtype=torch.float32)
        gru_state_dict['weight_hh_l0'] = torch.tensor(
            R_torch, dtype=torch.float32)
        gru_state_dict['bias_ih_l0'] = torch.tensor(
            bW_torch, dtype=torch.float32)
        gru_state_dict['bias_hh_l0'] = torch.tensor(
            bR_torch, dtype=torch.float32)

    # Assign reverse weights
    reverseEnabled = direction in [GRU_DIR_REVERSE, GRU_DIR_BIDIRECTIONAL]
    if reverseEnabled:
        if direction == GRU_DIR_REVERSE:
            reverse_dir_idx = 0
            (W_torch, R_torch, bW_torch, bR_torch) = get_torch_weights(reverse_dir_idx)
            gru_state_dict['weight_ih_l0'] = torch.tensor(
                W_torch, dtype=torch.float32)
            gru_state_dict['weight_hh_l0'] = torch.tensor(
                R_torch, dtype=torch.float32)
            gru_state_dict['bias_ih_l0'] = torch.tensor(
                bW_torch, dtype=torch.float32)
            gru_state_dict['bias_hh_l0'] = torch.tensor(
                bR_torch, dtype=torch.float32)
        else:
            reverse_dir_idx = 1
            (W_torch, R_torch, bW_torch, bR_torch) = get_torch_weights(reverse_dir_idx)
            gru_state_dict['weight_ih_l0_reverse'] = torch.tensor(
                W_torch, dtype=torch.float32)
            gru_state_dict['weight_hh_l0_reverse'] = torch.tensor(
                R_torch, dtype=torch.float32)
            gru_state_dict['bias_ih_l0_reverse'] = torch.tensor(
                bW_torch, dtype=torch.float32)
            gru_state_dict['bias_hh_l0_reverse'] = torch.tensor(
                bR_torch, dtype=torch.float32)

    # Set GRU state dictionary
    gru.load_state_dict(gru_state_dict, strict=True)

    # Perform inference
    X_torch = torch.tensor(X, dtype=torch.float32)
    initial_h_torch = torch.tensor(initial_h, dtype=torch.float32)
    if direction == GRU_DIR_REVERSE:
        Y, next_h = gru(X_torch.flip([0]), initial_h_torch)
        Y = Y.flip([0])
    else:
        Y, next_h = gru(X_torch, initial_h_torch)

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

    # Function to compute one GRU cell
    def compute_gru(forward):
        dir_idx = 0 if forward else (0 if direction == GRU_DIR_REVERSE else 1)
        Wz, Wr, Wh, Rz, Rr, Rh, bWz, bWr, bWh, bRz, bRr, bRh = get_weights(
            dir_idx)

        def f(x): return (1 / (1 + np.exp(-x)))

        def g(x): return np.tanh(x)

        def mm(x, w): return np.matmul(x, w.transpose())
        Ht = np.reshape(initial_h[dir_idx, :, :], [batch_size, hidden_size])

        Yslices = list()
        for t in range(seq_length):
            xt = Xslices[t] if forward else Xslices[seq_length - 1 - t]
            zt = f(mm(xt, Wz) + bWz + mm(Ht, Rz) + bRz)
            rt = f(mm(xt, Wr) + bWr + mm(Ht, Rr) + bRr)
            if linear_before_reset:
                htild = g(mm(xt, Wh) + bWh + rt * (mm(Ht, Rh) + bRh))
            else:
                htild = g(mm(xt, Wh) + bWh + mm(rt * Ht, Rh) + bRh)
            Ht = (1 - zt) * htild + zt * Ht
            Yslices.append(Ht)
        return Yslices, Ht

    Yslices = list()
    Hslices = list()

    # Compute forward GRU
    forwardYslices = list()
    if forwardEnabled:
        Yt, Ht = compute_gru(True)
        forwardYslices += Yt
        Hslices.append(Ht)

    # Compute reverse GRU
    reverseYslices = list()
    if reverseEnabled:
        Yt, Ht = compute_gru(False)
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

    # Use numpy implementation when linear_before_reset = False, else assert errors
    if linear_before_reset is False:
        Y_ref = Y_ref_np
        Y_h_ref = Y_h_ref_np
    else:
        assert np.max(np.abs(Y_ref - Y_ref_np)
                      ) < 1e-6, "Mismatch between Pytorch and Numpy GRU implementation"
        assert np.max(np.abs(Y_h_ref - Y_h_ref_np)
                      ) < 1e-6, "Mismatch between Pytorch and Numpy GRU implementation"

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

    # GRU node definition
    gru_node_def = onnx.helper.make_node(
        'GRU',
        name='gru',
        inputs=node_inputs,
        outputs=node_outputs,
        hidden_size=hidden_size,
        direction=direction,
        linear_before_reset=linear_before_reset
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

    # GRU inputs
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

    # GRU initializers
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
    graph_name = 'gru_test'
    graph_def = helper.make_graph(
        [gru_node_def, err_node_def], graph_name, inputs=graph_input, outputs=graph_output)

    # Set initializers
    graph_def.initializer.extend(graph_init)

    # --------------------------------------------- MODEL DEFINITION  --------------------------------------------------
    # Define model (ModelProto)
    model_def = helper.make_model(graph_def, producer_name='onnx-gru')

    # Check model
    onnx.checker.check_model(model_def)

    # Print model
    with open(model_path, 'w') as f:
        f.write(str(model_def))


# Forward GRU
gen_gru_onnx_test_model(model_path='gruForward.onnxtxt',
                        seq_length=2,
                        batch_size=5,
                        hidden_size=4,
                        input_size=3,
                        direction='forward',
                        has_bias=True,
                        has_sequence_lens=False,
                        has_initial_h=True,
                        linear_before_reset=False)

# Reverse GRU
gen_gru_onnx_test_model(model_path='gruReverse.onnxtxt',
                        seq_length=2,
                        batch_size=5,
                        hidden_size=4,
                        input_size=3,
                        direction='reverse',
                        has_bias=True,
                        has_sequence_lens=False,
                        has_initial_h=True,
                        linear_before_reset=False)

# Bidirectional GRU
gen_gru_onnx_test_model(model_path='gruBidirectional.onnxtxt',
                        seq_length=2,
                        batch_size=5,
                        hidden_size=4,
                        input_size=3,
                        direction='bidirectional',
                        has_bias=True,
                        has_sequence_lens=False,
                        has_initial_h=True,
                        linear_before_reset=False)

# Forward no bias GRU
gen_gru_onnx_test_model(model_path='gruForwardNoBias.onnxtxt',
                        seq_length=1,
                        batch_size=5,
                        hidden_size=4,
                        input_size=3,
                        direction='forward',
                        has_bias=False,
                        has_sequence_lens=False,
                        has_initial_h=True,
                        linear_before_reset=False)

# Forward no state GRU
gen_gru_onnx_test_model(model_path='gruForwardNoState.onnxtxt',
                        seq_length=1,
                        batch_size=5,
                        hidden_size=4,
                        input_size=3,
                        direction='forward',
                        has_bias=True,
                        has_sequence_lens=False,
                        has_initial_h=False,
                        linear_before_reset=False)

# Forward with linear before reset
gen_gru_onnx_test_model(model_path='gruForwardLinearBeforeReset.onnxtxt',
                        seq_length=1,
                        batch_size=5,
                        hidden_size=4,
                        input_size=3,
                        direction='forward',
                        has_bias=True,
                        has_sequence_lens=False,
                        has_initial_h=True,
                        linear_before_reset=True)
