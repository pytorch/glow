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


# Function to generate AudioSpectrogram ONNX test model.
def gen_spectrogram_onnx_test_model(model_path, window_count, window_size, stride, magnitude_squared=True):

    # Tensor sizes.
    input_length = window_size + (window_count - 1) * stride
    fft_length = int(2 ** np.ceil(np.log2(window_size)))
    input_shape = [1, input_length]
    spectrogram_length = int(fft_length / 2 + 1)
    spectrogram_shape = [window_count, spectrogram_length]

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
                                                 magnitude_squared=magnitude_squared)

    # Run TensorFlow model and get reference output.
    with tf.Session() as sess:
        spectrogram_ref = sess.run(tf_spectrogram)
    spectrogram_ref = np.reshape(spectrogram_ref, spectrogram_shape)

    # ---------------------------------------------- NODE DEFINITION  --------------------------------------------------
    # AudioSpectrogram node definition.
    spectrogram_node_def = onnx.helper.make_node(
        'AudioSpectrogram',
        name='audio_spectrogram',
        inputs=['input'],
        outputs=['spectrogram'],
        window_size=int(window_size),
        stride=int(stride),
        magnitude_squared=int(magnitude_squared)
    )

    # Error node definition.
    err_node_def = onnx.helper.make_node(
        'Sub',
        name='error',
        inputs=['spectrogram', 'spectrogram_ref'],
        outputs=['spectrogram_err']
    )

    # --------------------------------------------- GRAPH DEFINITION  --------------------------------------------------
    graph_input = list()
    graph_init = list()
    graph_output = list()

    # Graph inputs.
    graph_input.append(helper.make_tensor_value_info(
        'input', TensorProto.FLOAT, input_shape))
    graph_input.append(helper.make_tensor_value_info(
        'spectrogram_ref', TensorProto.FLOAT, spectrogram_shape))

    # Graph initializers.
    graph_init.append(make_init('input', TensorProto.FLOAT, input_data))
    graph_init.append(make_init('spectrogram_ref',
                                TensorProto.FLOAT, spectrogram_ref))

    # Graph outputs.
    graph_output.append(helper.make_tensor_value_info(
        'spectrogram_err', TensorProto.FLOAT, spectrogram_shape))

    # Graph name.
    graph_name = 'audio_spectrogram_test'

    # Define graph (GraphProto).
    graph_def = helper.make_graph([spectrogram_node_def, err_node_def], graph_name, inputs=graph_input,
                                  outputs=graph_output)

    # Set initializers.
    graph_def.initializer.extend(graph_init)

    # --------------------------------------------- MODEL DEFINITION  --------------------------------------------------
    # Define model (ModelProto).
    model_def = helper.make_model(
        graph_def, producer_name='onnx-audio-spectrogram')

    # Print model.
    with open(model_path, 'w') as f:
        f.write(str(model_def))


# One window spectrogram.
gen_spectrogram_onnx_test_model(model_path='audioSpectrogramOneWindow.onnxtxt',
                                window_count=1,
                                window_size=512,
                                stride=256,
                                magnitude_squared=True)

# Two window spectrogram.
gen_spectrogram_onnx_test_model(model_path='audioSpectrogramTwoWindow.onnxtxt',
                                window_count=2,
                                window_size=640,
                                stride=320,
                                magnitude_squared=True)

# Magnitude non-squared.
gen_spectrogram_onnx_test_model(model_path='audioSpectrogramNonSquared.onnxtxt',
                                window_count=1,
                                window_size=640,
                                stride=320,
                                magnitude_squared=False)
