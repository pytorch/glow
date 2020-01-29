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

# LSTM enums
LSTM_DIR_FORWARD = 'forward'
LSTM_DIR_REVERSE = 'reverse'
LSTM_DIR_BIDIRECTIONAL = 'bidirectional'
LSTM_DIRS = [LSTM_DIR_FORWARD, LSTM_DIR_REVERSE, LSTM_DIR_BIDIRECTIONAL]


# ONNX utility
def make_init(name, type, tensor):
    return helper.make_tensor(name=name, data_type=type, dims=tensor.shape, vals=tensor.reshape(tensor.size).tolist())


# Function to generate LSTM ONNX test model
def gen_lstm_onnx_test_model(model_path, seq_length, batch_size, hidden_size, input_size, direction, has_bias,
                             has_sequence_lens, has_initial_h, has_initial_c, has_peephole, input_forget=False):

    # Validate parameters
    assert direction in LSTM_DIRS, 'ONNX LSTM direction invalid!'
    assert not has_sequence_lens, 'ONNX LSTM Variable sequence length not supported'

    # Get number of directions
    num_directions = 2 if (direction == LSTM_DIR_BIDIRECTIONAL) else 1

    # Tensor sizes
    X_shape = [seq_length, batch_size, input_size]
    W_shape = [num_directions, 4 * hidden_size, input_size]
    R_shape = [num_directions, 4 * hidden_size, hidden_size]
    B_shape = [num_directions, 8 * hidden_size]
    sequence_lens_shape = [batch_size]
    initial_h_shape = [num_directions, batch_size, hidden_size]
    initial_c_shape = [num_directions, batch_size, hidden_size]
    P_shape = [num_directions, 3 * hidden_size]
    Y_shape = [seq_length, num_directions, batch_size, hidden_size]

    # Generate random inputs (weights are assumed concatenated in ONNX format: i,o,f,c)
    np.random.seed(1)
    X = np.random.randn(*X_shape)
    W = np.random.randn(*W_shape)
    R = np.random.randn(*R_shape)
    B = np.random.randn(*B_shape) if has_bias else np.zeros(B_shape)
    sequence_lens = np.random.randint(
        1, seq_length, batch_size) if has_sequence_lens else np.tile(seq_length, batch_size)
    initial_h = np.random.randn(
        *initial_h_shape) if has_initial_h else np.zeros(initial_h_shape)
    initial_c = np.random.randn(
        *initial_c_shape) if has_initial_c else np.zeros(initial_c_shape)
    P = np.random.randn(*P_shape) if has_peephole else np.zeros(P_shape)

    # Function to get all the weight components for the given direction
    def get_weights(dir_idx):
        Wi = np.reshape(W[dir_idx, 0 * hidden_size: 1 *
                          hidden_size, :], [hidden_size, input_size])
        Wo = np.reshape(W[dir_idx, 1 * hidden_size: 2 *
                          hidden_size, :], [hidden_size, input_size])
        Wf = np.reshape(W[dir_idx, 2 * hidden_size: 3 *
                          hidden_size, :], [hidden_size, input_size])
        Wc = np.reshape(W[dir_idx, 3 * hidden_size: 4 *
                          hidden_size, :], [hidden_size, input_size])
        Ri = np.reshape(R[dir_idx, 0 * hidden_size: 1 *
                          hidden_size, :], [hidden_size, hidden_size])
        Ro = np.reshape(R[dir_idx, 1 * hidden_size: 2 *
                          hidden_size, :], [hidden_size, hidden_size])
        Rf = np.reshape(R[dir_idx, 2 * hidden_size: 3 *
                          hidden_size, :], [hidden_size, hidden_size])
        Rc = np.reshape(R[dir_idx, 3 * hidden_size: 4 *
                          hidden_size, :], [hidden_size, hidden_size])
        bWi = np.reshape(B[dir_idx, 0 * hidden_size: 1 *
                           hidden_size], [hidden_size])
        bWo = np.reshape(B[dir_idx, 1 * hidden_size: 2 *
                           hidden_size], [hidden_size])
        bWf = np.reshape(B[dir_idx, 2 * hidden_size: 3 *
                           hidden_size], [hidden_size])
        bWc = np.reshape(B[dir_idx, 3 * hidden_size: 4 *
                           hidden_size], [hidden_size])
        bRi = np.reshape(B[dir_idx, 4 * hidden_size: 5 *
                           hidden_size], [hidden_size])
        bRo = np.reshape(B[dir_idx, 5 * hidden_size: 6 *
                           hidden_size], [hidden_size])
        bRf = np.reshape(B[dir_idx, 6 * hidden_size: 7 *
                           hidden_size], [hidden_size])
        bRc = np.reshape(B[dir_idx, 7 * hidden_size: 8 *
                           hidden_size], [hidden_size])
        Pi = np.tile(P[dir_idx, 0 * hidden_size: 1 *
                       hidden_size], (batch_size, 1))
        Po = np.tile(P[dir_idx, 1 * hidden_size: 2 *
                       hidden_size], (batch_size, 1))
        Pf = np.tile(P[dir_idx, 2 * hidden_size: 3 *
                       hidden_size], (batch_size, 1))
        return Wi, Wo, Wf, Wc, Ri, Ro, Rf, Rc, bWi, bWo, bWf, bWc, bRi, bRo, bRf, bRc, Pi, Po, Pf

    # Function to get PyTorch weights (which are in the i, f, c, o order)
    def get_torch_weights(dir_idx):
        Wi, Wo, Wf, Wc, Ri, Ro, Rf, Rc, bWi, bWo, bWf, bWc, bRi, bRo, bRf, bRc, Pi, Po, Pf = get_weights(
            dir_idx)
        W_torch = np.concatenate((Wi, Wf, Wc, Wo), 0)
        R_torch = np.concatenate((Ri, Rf, Rc, Ro), 0)
        bW_torch = np.concatenate((bWi, bWf, bWc, bWo), 0)
        bR_torch = np.concatenate((bRi, bRf, bRc, bRo), 0)
        return (W_torch, R_torch, bW_torch, bR_torch)

    # ----------------------------------------- COMPUTE pyTORCH REFERENCE ----------------------------------------------
    # Compute reference using Pytorch. Pytorch LSTM has only forward/bidirectional so we will do the reverse LSTM using
    # a Pytorch forward LSTM.
    lstm = torch.nn.LSTM(input_size=input_size,
                         hidden_size=hidden_size,
                         num_layers=1,
                         bias=True,
                         batch_first=False,
                         dropout=0,
                         bidirectional=(direction == LSTM_DIR_BIDIRECTIONAL))

    # Get LSTM state dictionary
    lstm_state_dict = lstm.state_dict()

    # Assign forward weights
    forwardEnabled = direction in [LSTM_DIR_FORWARD, LSTM_DIR_BIDIRECTIONAL]
    if forwardEnabled:
        forward_dir_idx = 0
        (W_torch, R_torch, bW_torch, bR_torch) = get_torch_weights(forward_dir_idx)
        lstm_state_dict['weight_ih_l0'] = torch.tensor(
            W_torch, dtype=torch.float32)
        lstm_state_dict['weight_hh_l0'] = torch.tensor(
            R_torch, dtype=torch.float32)
        lstm_state_dict['bias_ih_l0'] = torch.tensor(
            bW_torch, dtype=torch.float32)
        lstm_state_dict['bias_hh_l0'] = torch.tensor(
            bR_torch, dtype=torch.float32)

    # Assign reverse weights
    reverseEnabled = direction in [LSTM_DIR_REVERSE, LSTM_DIR_BIDIRECTIONAL]
    if reverseEnabled:
        if direction == LSTM_DIR_REVERSE:
            reverse_dir_idx = 0
            (W_torch, R_torch, bW_torch, bR_torch) = get_torch_weights(reverse_dir_idx)
            lstm_state_dict['weight_ih_l0'] = torch.tensor(
                W_torch, dtype=torch.float32)
            lstm_state_dict['weight_hh_l0'] = torch.tensor(
                R_torch, dtype=torch.float32)
            lstm_state_dict['bias_ih_l0'] = torch.tensor(
                bW_torch, dtype=torch.float32)
            lstm_state_dict['bias_hh_l0'] = torch.tensor(
                bR_torch, dtype=torch.float32)
        else:
            reverse_dir_idx = 1
            (W_torch, R_torch, bW_torch, bR_torch) = get_torch_weights(reverse_dir_idx)
            lstm_state_dict['weight_ih_l0_reverse'] = torch.tensor(
                W_torch, dtype=torch.float32)
            lstm_state_dict['weight_hh_l0_reverse'] = torch.tensor(
                R_torch, dtype=torch.float32)
            lstm_state_dict['bias_ih_l0_reverse'] = torch.tensor(
                bW_torch, dtype=torch.float32)
            lstm_state_dict['bias_hh_l0_reverse'] = torch.tensor(
                bR_torch, dtype=torch.float32)

    # Set LSTM state dictionary
    lstm.load_state_dict(lstm_state_dict, strict=True)

    # Perform inference
    X_torch = torch.tensor(X, dtype=torch.float32)
    initial_h_torch = torch.tensor(initial_h, dtype=torch.float32)
    initial_c_torch = torch.tensor(initial_c, dtype=torch.float32)
    if direction == LSTM_DIR_REVERSE:
        Y, (next_h, next_c) = lstm(X_torch.flip(
            [0]), (initial_h_torch, initial_c_torch))
        Y = Y.flip([0])
    else:
        Y, (next_h, next_c) = lstm(X_torch, (initial_h_torch, initial_c_torch))

    # Reshape output to ONNX format [seq_length, num_directions, batch_size, hidden_size]
    Y_ref = Y.detach().numpy()
    Y_ref = np.reshape(
        Y_ref, [seq_length, batch_size, num_directions, hidden_size])
    Y_ref = np.transpose(Y_ref, [0, 2, 1, 3])

    # Reshape states to ONNX format
    Y_h_ref = next_h.detach().numpy()
    Y_c_ref = next_c.detach().numpy()

    # --------------------------------------- COMPUTE PYTHON-NUMPY REFERENCE -------------------------------------------
    # Create X slices
    Xslices = list()
    for t in range(seq_length):
        Xslices.append(np.reshape(X[t, :, :], [batch_size, input_size]))

    # Function to compute one LSTM cell
    def compute_lstm(forward):
        dir_idx = 0 if forward else (0 if direction == LSTM_DIR_REVERSE else 1)
        Wi, Wo, Wf, Wc, Ri, Ro, Rf, Rc, bWi, bWo, bWf, bWc, bRi, bRo, bRf, bRc, Pi, Po, Pf = get_weights(
            dir_idx)

        def f(x): return (1 / (1 + np.exp(-x)))

        def g(x): return np.tanh(x)

        def h(x): return np.tanh(x)

        def mm(x, w): return np.matmul(x, w.transpose())
        Ht = np.reshape(initial_h[dir_idx, :, :], [batch_size, hidden_size])
        Ct = np.reshape(initial_c[dir_idx, :, :], [batch_size, hidden_size])

        Yslices = list()
        for t in range(seq_length):
            xt = Xslices[t] if forward else Xslices[seq_length - 1 - t]
            ft = f(mm(xt, Wf) + bWf + mm(Ht, Rf) + bRf + Pf * Ct)
            if input_forget:
                it = 1 - ft
            else:
                it = f(mm(xt, Wi) + bWi + mm(Ht, Ri) + bRi + Pi * Ct)
            ctild = g(mm(xt, Wc) + bWc + mm(Ht, Rc) + bRc)
            Ct = ft * Ct + it * ctild
            ot = f(mm(xt, Wo) + bWo + mm(Ht, Ro) + bRo + Po * Ct)
            Ht = ot * h(Ct)
            Yslices.append(Ht)
        return Yslices, Ht, Ct

    Yslices = list()
    Hslices = list()
    Cslices = list()

    # Compute forward LSTM
    forwardYslices = list()
    if forwardEnabled:
        Yt, Ht, Ct = compute_lstm(True)
        forwardYslices += Yt
        Hslices.append(Ht)
        Cslices.append(Ct)

    # Compute reverse LSTM
    reverseYslices = list()
    if reverseEnabled:
        Yt, Ht, Ct = compute_lstm(False)
        reverseYslices += Yt
        Hslices.append(Ht)
        Cslices.append(Ct)

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
    Y_c_ref_np = np.concatenate(Cslices, 0).reshape(
        [num_directions, batch_size, hidden_size])

    # Use numpy implementation when using peepholes or input_forget, else assert errors
    if has_peephole or input_forget:
        Y_ref = Y_ref_np
        Y_h_ref = Y_h_ref_np
        Y_c_ref = Y_c_ref_np
    else:
        assert np.max(np.abs(Y_ref - Y_ref_np)
                      ) < 1e-6, "Mismatch between Pytorch and Numpy LSTM implementation"
        assert np.max(np.abs(Y_h_ref - Y_h_ref_np)
                      ) < 1e-6, "Mismatch between Pytorch and Numpy LSTM implementation"
        assert np.max(np.abs(Y_c_ref - Y_c_ref_np)
                      ) < 1e-6, "Mismatch between Pytorch and Numpy LSTM implementation"

    # ---------------------------------------------- NODE DEFINITION  --------------------------------------------------
    # Node inputs
    node_inputs = ['X',
                   'W',
                   'R',
                   'B' if has_bias else '',
                   '',
                   'initial_h' if has_initial_h else '',
                   'initial_c' if has_initial_c else '',
                   'P' if has_peephole else '']

    # Node outputs
    node_outputs = ['Y', 'Y_h', 'Y_c']

    # LSTM node definition
    lstm_node_def = onnx.helper.make_node(
        'LSTM',
        name='lstm',
        inputs=node_inputs,
        outputs=node_outputs,
        hidden_size=hidden_size,
        direction=direction,
        input_forget=input_forget
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

    # LSTM inputs
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
    if has_initial_c:
        graph_input.append(helper.make_tensor_value_info(
            'initial_c', TensorProto.FLOAT, initial_c_shape))
    if has_peephole:
        graph_input.append(helper.make_tensor_value_info(
            'P', TensorProto.FLOAT, P_shape))

    # Reference input
    graph_input.append(helper.make_tensor_value_info(
        'Y_ref', TensorProto.FLOAT, Y_shape))

    # LSTM initializers
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
    if has_initial_c:
        graph_init.append(make_init('initial_c', TensorProto.FLOAT, initial_c))
    if has_peephole:
        graph_init.append(make_init('P', TensorProto.FLOAT, P))

    # Reference initializer
    graph_init.append(make_init('Y_ref', TensorProto.FLOAT, Y_ref))

    # Graph outputs
    graph_output.append(helper.make_tensor_value_info(
        'Y_err', TensorProto.FLOAT, Y_shape))

    # Define graph (GraphProto)
    graph_name = 'lstm_test'
    graph_def = helper.make_graph(
        [lstm_node_def, err_node_def], graph_name, inputs=graph_input, outputs=graph_output)

    # Set initializers
    graph_def.initializer.extend(graph_init)

    # --------------------------------------------- MODEL DEFINITION  --------------------------------------------------
    # Define model (ModelProto)
    model_def = helper.make_model(graph_def, producer_name='onnx-lstm')

    # Check model
    onnx.checker.check_model(model_def)

    # Print model
    with open(model_path, 'w') as f:
        f.write(str(model_def))


# Forward LSTM
gen_lstm_onnx_test_model(model_path='lstmForward.onnxtxt',
                         seq_length=2,
                         batch_size=5,
                         hidden_size=4,
                         input_size=3,
                         direction='forward',
                         has_bias=True,
                         has_sequence_lens=False,
                         has_initial_h=True,
                         has_initial_c=True,
                         has_peephole=False,
                         input_forget=False)

# Reverse LSTM
gen_lstm_onnx_test_model(model_path='lstmReverse.onnxtxt',
                         seq_length=2,
                         batch_size=5,
                         hidden_size=4,
                         input_size=3,
                         direction='reverse',
                         has_bias=True,
                         has_sequence_lens=False,
                         has_initial_h=True,
                         has_initial_c=True,
                         has_peephole=False,
                         input_forget=False)

# Bidirectional LSTM
gen_lstm_onnx_test_model(model_path='lstmBidirectional.onnxtxt',
                         seq_length=2,
                         batch_size=5,
                         hidden_size=4,
                         input_size=3,
                         direction='bidirectional',
                         has_bias=True,
                         has_sequence_lens=False,
                         has_initial_h=True,
                         has_initial_c=True,
                         has_peephole=False,
                         input_forget=False)

# Forward no bias LSTM
gen_lstm_onnx_test_model(model_path='lstmForwardNoBias.onnxtxt',
                         seq_length=1,
                         batch_size=5,
                         hidden_size=4,
                         input_size=3,
                         direction='forward',
                         has_bias=False,
                         has_sequence_lens=False,
                         has_initial_h=True,
                         has_initial_c=True,
                         has_peephole=False,
                         input_forget=False)

# Forward no state LSTM
gen_lstm_onnx_test_model(model_path='lstmForwardNoState.onnxtxt',
                         seq_length=1,
                         batch_size=5,
                         hidden_size=4,
                         input_size=3,
                         direction='forward',
                         has_bias=True,
                         has_sequence_lens=False,
                         has_initial_h=False,
                         has_initial_c=False,
                         has_peephole=False,
                         input_forget=False)

# Forward with peephole LSTM
gen_lstm_onnx_test_model(model_path='lstmForwardWithPeephole.onnxtxt',
                         seq_length=1,
                         batch_size=5,
                         hidden_size=4,
                         input_size=3,
                         direction='forward',
                         has_bias=True,
                         has_sequence_lens=False,
                         has_initial_h=True,
                         has_initial_c=True,
                         has_peephole=True,
                         input_forget=False)

# Forward with input forget LSTM
gen_lstm_onnx_test_model(model_path='lstmForwardInputForget.onnxtxt',
                         seq_length=1,
                         batch_size=5,
                         hidden_size=4,
                         input_size=3,
                         direction='forward',
                         has_bias=True,
                         has_sequence_lens=False,
                         has_initial_h=True,
                         has_initial_c=True,
                         has_peephole=False,
                         input_forget=True)
