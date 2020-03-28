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
import tensorflow as tf
from tensorflow.python.ops import gen_audio_ops as audio_ops


# ONNX utility.
def make_init(name, dtype, tensor):
    return helper.make_tensor(name=name, data_type=dtype, dims=tensor.shape, vals=tensor.reshape(tensor.size).tolist())


# Function to generate MFCC ONNX test model.
def gen_mfcc_onnx_test_model(model_path, window_count, window_size, stride, sample_rate, lower_frequency_limit,
                             upper_frequency_limit, filterbank_channel_count, dct_coefficient_count):

    # Tensor sizes.
    input_length = window_size + (window_count - 1) * stride
    fft_length = int(2 ** np.ceil(np.log2(window_size)))
    input_shape = [1, input_length]
    spectrogram_length = int(fft_length / 2 + 1)
    spectrogram_shape = [window_count, spectrogram_length]
    coefficients_shape = [window_count, dct_coefficient_count]

    # Generate random input data.
    np.random.seed(1)
    input_data = np.random.randn(*input_shape)

    # ----------------------------------------- COMPUTE TensorFlow REFERENCE -------------------------------------------
    # Define TensorFlow model.
    tf_input = tf.constant(input_data.reshape(
        [input_length, 1]), name='input', dtype=tf.float32)
    tf_spectrogram = audio_ops.audio_spectrogram(tf_input,
                                                 window_size=window_size,
                                                 stride=stride,
                                                 magnitude_squared=True)
    tf_mfcc = audio_ops.mfcc(spectrogram=tf_spectrogram,
                             sample_rate=sample_rate,
                             upper_frequency_limit=upper_frequency_limit,
                             lower_frequency_limit=lower_frequency_limit,
                             filterbank_channel_count=filterbank_channel_count,
                             dct_coefficient_count=dct_coefficient_count)

    # Run TensorFlow model and get spectrogram input.
    with tf.Session() as sess:
        spectrogram = sess.run(tf_spectrogram)
    spectrogram = np.reshape(spectrogram, spectrogram_shape)

    # Run TensorFlow model and get reference output coefficients.
    with tf.Session() as sess:
        coefficients_ref = sess.run(tf_mfcc)
    coefficients_ref = np.reshape(coefficients_ref, coefficients_shape)

    # ---------------------------------------------- NODE DEFINITION  --------------------------------------------------
    # MFCC node definition.
    mfcc_node_def = onnx.helper.make_node(
        'MFCC',
        name='mfcc',
        inputs=['spectrogram'],
        outputs=['coefficients'],
        sample_rate=float(sample_rate),
        lower_frequency_limit=float(lower_frequency_limit),
        upper_frequency_limit=float(upper_frequency_limit),
        filterbank_channel_count=int(filterbank_channel_count),
        dct_coefficient_count=int(dct_coefficient_count)
    )

    # Error node definition.
    err_node_def = onnx.helper.make_node(
        'Sub',
        name='error',
        inputs=['coefficients', 'coefficients_ref'],
        outputs=['coefficients_err']
    )

    # --------------------------------------------- GRAPH DEFINITION  --------------------------------------------------
    graph_input = list()
    graph_init = list()
    graph_output = list()

    # Graph inputs.
    graph_input.append(helper.make_tensor_value_info(
        'spectrogram', TensorProto.FLOAT, spectrogram_shape))
    graph_input.append(helper.make_tensor_value_info(
        'coefficients_ref', TensorProto.FLOAT, coefficients_shape))

    # Graph initializers.
    graph_init.append(make_init('spectrogram', TensorProto.FLOAT, spectrogram))
    graph_init.append(make_init('coefficients_ref',
                                TensorProto.FLOAT, coefficients_ref))

    # Graph outputs.
    graph_output.append(helper.make_tensor_value_info(
        'coefficients_err', TensorProto.FLOAT, coefficients_shape))

    # Graph name.
    graph_name = 'mfcc_test'

    # Define graph (GraphProto).
    graph_def = helper.make_graph(
        [mfcc_node_def, err_node_def], graph_name, inputs=graph_input, outputs=graph_output)

    # Set initializers.
    graph_def.initializer.extend(graph_init)

    # --------------------------------------------- MODEL DEFINITION  --------------------------------------------------
    # Define model (ModelProto).
    model_def = helper.make_model(graph_def, producer_name='onnx-mfcc')

    # Print model.
    with open(model_path, 'w') as f:
        f.write(str(model_def))


# One window MFCC.
gen_mfcc_onnx_test_model(model_path='mfccOneWindow.onnxtxt',
                         window_count=1,
                         window_size=640,
                         stride=320,
                         sample_rate=16000,
                         lower_frequency_limit=20,
                         upper_frequency_limit=4000,
                         filterbank_channel_count=40,
                         dct_coefficient_count=10)

# Two window MFCC.
gen_mfcc_onnx_test_model(model_path='mfccTwoWindow.onnxtxt',
                         window_count=2,
                         window_size=512,
                         stride=256,
                         sample_rate=16000,
                         lower_frequency_limit=20,
                         upper_frequency_limit=4000,
                         filterbank_channel_count=40,
                         dct_coefficient_count=10)
