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

# This is a helper script that generates the TensorFlowLite models used
# for testing the Glow importer. The models will be generated in a local
# folder here 'tflite_models'. In order for the models to be used for unit
# testing the files must be copied in the folder:
#     'glow\tests\models\tfliteModels'
# To generate the models you need to run this script without arguments:
#     python gen_tflite_models.py
# Python requirements: Python 3.6
# Python package requirements:
#     TensorFlow 2.1.0
#     Keras 2.3.1
#     Numpy 1.16.2
#     shutil, os, other dependencies

import os
import shutil

import numpy as np
import tensorflow as tf
import tensorflow.keras.backend as keras_backend
import tensorflow.keras.layers as layers
from tensorflow.keras.models import Model
from tensorflow.python.tools import freeze_graph


# ----------------------------------------------------------------------------------------------------------------------
#                                                      UTILS
# ----------------------------------------------------------------------------------------------------------------------
# Temporary folder path.
TEMP_DIR = os.path.join(os.path.dirname(__file__), "temp")

# Output model folder.
OUT_DIR = os.path.join(os.path.dirname(__file__), "tflite_models")


# Clean temporary directory.
def clean_dir(dir_path):
    if os.path.exists(dir_path):
        shutil.rmtree(dir_path)
    os.mkdir(dir_path)


# Remove temporary directory.
def rm_dir(dir_path):
    if os.path.exists(dir_path):
        shutil.rmtree(dir_path)


# Function to save a model in TensorFlowLite format.
def save_model(model, filename):
    # Print status.
    print('Saving model "%s" ...' % filename)

    # Clean temporary folder.
    clean_dir(TEMP_DIR)

    # Get model inputs.
    model_inputs_dict = dict()
    model_inputs_array = []
    for idx in range(len(model.inputs)):
        model_inputs_dict["input_%d" % idx] = model.inputs[idx]
        model_inputs_array.append(model.inputs[idx].op.name)

    # Get model outputs.
    model_outputs_dict = dict()
    model_outputs_array = []
    for idx in range(len(model.outputs)):
        if idx == 0:
            output_name = model.outputs[idx].op.name
        else:
            output_name = model.outputs[idx].name
        model_outputs_dict[output_name] = model.outputs[idx]
        model_outputs_array.append(output_name)

    # Save TensorFlow checkpoint.
    tf.saved_model.simple_save(
        keras_backend.get_session(),
        os.path.join(TEMP_DIR, "checkpoint"),
        inputs=model_inputs_dict,
        outputs=model_outputs_dict,
    )

    # Freeze TensorFlow graph.
    freeze_graph.freeze_graph(
        None,
        None,
        None,
        None,
        model.outputs[0].op.name,
        None,
        None,
        os.path.join(TEMP_DIR, "model.pb"),
        False,
        "",
        input_saved_model_dir=os.path.join(TEMP_DIR, "checkpoint"),
    )

    # Convert and save TensorFlowLite model.
    converter = tf.lite.TFLiteConverter.from_frozen_graph(
        os.path.join(TEMP_DIR, "model.pb"),
        input_arrays=model_inputs_array,
        output_arrays=model_outputs_array,
    )
    converter.dump_graphviz_video = False
    converter.allow_custom_ops = True
    tflite_model = converter.convert()
    model_filename = os.path.join(OUT_DIR, filename)
    if not model_filename.endswith(".tflite"):
        model_filename += ".tflite"
    open(model_filename, "wb").write(tflite_model)

    # Clean temporary folder.
    rm_dir(TEMP_DIR)


# Function to save a tensor in binary format. In order for the GIT system
# to correctly recognize these files as binary we will add a leading '0'
# byte into the file.
def save_tensor(tensor, filename):
    byte_array = b"\x00" + tensor.tobytes(order="C")
    with open(os.path.join(OUT_DIR, filename), "wb") as fh:
        fh.write(byte_array)


# Create output directory.
clean_dir(OUT_DIR)


# ----------------------------------------------------------------------------------------------------------------------
#                                                        Strided Slice
# ----------------------------------------------------------------------------------------------------------------------
def gen_strided_slice(
    name,
    input_shape,
    begin,
    end,
    strides,
    begin_mask=0,
    end_mask=0,
    ellipsis_mask=0,
    new_axis_mask=0,
    shrink_axis_mask=0,
):
    # Create model.
    inp = layers.Input(
        name="input", batch_size=input_shape[0], shape=input_shape[1:], dtype=tf.float32
    )
    out = tf.strided_slice(
        inp,
        begin,
        end,
        strides,
        begin_mask,
        end_mask,
        ellipsis_mask,
        new_axis_mask,
        shrink_axis_mask,
    )
    model = Model(inputs=[inp], outputs=[out])
    # Create data.
    np.random.seed(0)
    inp_tensor = np.random.rand(*input_shape).astype(np.float32)
    out_tensor = model.predict([inp_tensor])
    # Save model.
    save_model(model, name)
    # Save data.
    save_tensor(inp_tensor, name + ".inp0")
    save_tensor(out_tensor, name + ".out0")
    # Clear session.
    keras_backend.clear_session()


# Basic test. Default strides are 1.
gen_strided_slice(
    name="strided_slice_test0",
    input_shape=(1, 2, 3),
    begin=(0, 0, 0),
    end=(1, 1, 1),
    strides=(1, 1, 1),
)

# Test begin_mask. Ignore "begin" value for 2nd dimension and use value for maximum range.
gen_strided_slice(
    name="strided_slice_test1",
    input_shape=(1, 3, 4),
    begin=(0, 2, 3),
    end=(1, 3, 4),
    strides=(1, 1, 1),
    begin_mask=2,
)

# Test end_mask. Ignore "end" value for 2nd dimension and use value for maximum range.
gen_strided_slice(
    name="strided_slice_test2",
    input_shape=(1, 3, 4),
    begin=(0, 0, 0),
    end=(1, 1, 1),
    strides=(1, 1, 1),
    end_mask=2,
)

# Test begin_mask & end_mask. Ignore "begin"/"end" value for 2nd dimension and use values for maximum range.
gen_strided_slice(
    name="strided_slice_test3",
    input_shape=(1, 3, 4),
    begin=(0, 1, 1),
    end=(1, 2, 2),
    strides=(1, 1, 1),
    begin_mask=2,
    end_mask=2,
)

# Test ellipsis_mask. Test access pattern [0, ..., 0] where the ellipsis position is marked as 0's for begin/end.
gen_strided_slice(
    name="strided_slice_test4",
    input_shape=(1, 3, 4),
    begin=(0, 0, 0),
    end=(1, 0, 1),
    strides=(1, 1, 1),
    begin_mask=0,
    end_mask=0,
    ellipsis_mask=2,
)

# Test new_axis_mask.
gen_strided_slice(
    name="strided_slice_test5",
    input_shape=(1, 3, 4),
    begin=(0, 0, 0),
    end=(1, 2, 3),
    strides=(1, 1, 1),
    new_axis_mask=2,
)

# Test shrink_axis_mask.
gen_strided_slice(
    name="strided_slice_test6",
    input_shape=(1, 3, 4),
    begin=(0, 0, 0),
    end=(1, 2, 3),
    strides=(1, 1, 1),
    shrink_axis_mask=2,
)


# ----------------------------------------------------------------------------------------------------------------------
#                                                        Select
# ----------------------------------------------------------------------------------------------------------------------
def gen_select_test(name, input_shape):
    # Create model.
    cond = layers.Input(
        name="cond", batch_size=input_shape[0], shape=input_shape[1:], dtype=tf.bool
    )
    lhs = layers.Input(
        name="lhs", batch_size=input_shape[0], shape=input_shape[1:], dtype=tf.float32
    )
    rhs = layers.Input(
        name="rhs", batch_size=input_shape[0], shape=input_shape[1:], dtype=tf.float32
    )
    out = tf.where(cond, x=lhs, y=rhs)
    model = Model(inputs=[cond, lhs, rhs], outputs=[out])
    # Create data.
    np.random.seed(0)
    cond_tensor = np.random.randint(low=0, high=2, size=input_shape).astype(np.bool)
    lhs_tensor = np.random.rand(*input_shape).astype(np.float32)
    rhs_tensor = np.random.rand(*input_shape).astype(np.float32)
    out_tensor = model.predict([cond_tensor, lhs_tensor, rhs_tensor])
    # Save model.
    save_model(model, name)
    # Save data.
    save_tensor(cond_tensor, name + ".inp0")
    save_tensor(lhs_tensor, name + ".inp1")
    save_tensor(rhs_tensor, name + ".inp2")
    save_tensor(out_tensor, name + ".out0")
    # Clear session.
    keras_backend.clear_session()


gen_select_test(name="select", input_shape=(1, 2, 3))


# ----------------------------------------------------------------------------------------------------------------------
#                                                        LogSoftmax
# ----------------------------------------------------------------------------------------------------------------------
def gen_log_softmax_test(name, input_shape, axis):
    # Create model.
    inp = layers.Input(name="input", batch_size=input_shape[0], shape=input_shape[1:])
    out = tf.nn.log_softmax(inp, axis=axis)
    model = Model(inputs=[inp], outputs=[out])
    # Create data.
    np.random.seed(0)
    inp_tensor = np.random.rand(*input_shape).astype(np.float32)
    out_tensor = model.predict(inp_tensor)
    # Save model.
    save_model(model, name)
    # Save data.
    save_tensor(inp_tensor, name + ".inp0")
    save_tensor(out_tensor, name + ".out0")
    # Clear session.
    keras_backend.clear_session()


gen_log_softmax_test(name="log_softmax", input_shape=(1, 3), axis=-1)


# ----------------------------------------------------------------------------------------------------------------------
#                                                      GATHER ND
# ----------------------------------------------------------------------------------------------------------------------
def gen_gather_nd_test(name, data_shape, indices_shape):
    # Create model.
    data = layers.Input(
        name="data", batch_size=data_shape[0], shape=data_shape[1:], dtype=tf.float32
    )
    indices = layers.Input(
        name="indices",
        batch_size=indices_shape[0],
        shape=indices_shape[1:],
        dtype=tf.int32,
    )
    out = tf.gather_nd(data, indices, batch_dims=0)
    model = Model(inputs=[data, indices], outputs=[out])
    # Create data.
    np.random.seed(0)
    data_tensor = np.random.rand(*data_shape).astype(np.float32)
    indices_tensor = np.random.randint(
        low=0, high=data_shape, size=indices_shape
    ).astype(np.int32)
    out_tensor = model.predict([data_tensor, indices_tensor])
    # Save model.
    save_model(model, name)
    # Save data.
    save_tensor(data_tensor, name + ".inp0")
    save_tensor(indices_tensor, name + ".inp1")
    save_tensor(out_tensor, name + ".out0")
    # Clear session.
    keras_backend.clear_session()


gen_gather_nd_test(name="gather_nd", data_shape=(2, 3, 4), indices_shape=(2, 3))


# ----------------------------------------------------------------------------------------------------------------------
#                                                      GATHER
# ----------------------------------------------------------------------------------------------------------------------
def gen_gather_test(name, data_shape, indices_shape, axis):
    # Create model.
    data = layers.Input(
        name="data", batch_size=data_shape[0], shape=data_shape[1:], dtype=tf.float32
    )
    indices = layers.Input(
        name="indices",
        batch_size=indices_shape[0],
        shape=indices_shape[1:],
        dtype=tf.int32,
    )
    out = tf.gather(data, indices, axis=axis, batch_dims=0)
    model = Model(inputs=[data, indices], outputs=[out])
    # Create data.
    np.random.seed(0)
    data_tensor = np.random.rand(*data_shape).astype(np.float32)
    indices_tensor = np.random.randint(data_shape[axis], size=indices_shape).astype(
        np.int32
    )
    out_tensor = model.predict([data_tensor, indices_tensor])
    # Save model.
    save_model(model, name)
    # Save data.
    save_tensor(data_tensor, name + ".inp0")
    save_tensor(indices_tensor, name + ".inp1")
    save_tensor(out_tensor, name + ".out0")
    # Clear session.
    keras_backend.clear_session()


gen_gather_test(
    name="gather_axis0", data_shape=(1, 2, 3, 4), indices_shape=(1, 5), axis=0
)
gen_gather_test(
    name="gather_axis1", data_shape=(1, 2, 3, 4), indices_shape=(1, 5), axis=1
)


# ----------------------------------------------------------------------------------------------------------------------
#                                                          CAST
# ----------------------------------------------------------------------------------------------------------------------
def gen_cast_test(name, input_shape, dtype):
    # Create model.
    inp = layers.Input(name="input", batch_size=input_shape[0], shape=input_shape[1:])
    out = tf.cast(inp, dtype=dtype)
    model = Model(inputs=[inp], outputs=[out])
    # Create data.
    np.random.seed(0)
    inp_tensor = np.random.rand(*input_shape).astype(np.float32)
    out_tensor = model.predict(inp_tensor)
    # Save model.
    save_model(model, name)
    # Save data.
    save_tensor(inp_tensor, name + ".inp0")
    save_tensor(out_tensor, name + ".out0")
    # Clear session.
    keras_backend.clear_session()


gen_cast_test(name="cast_f32_to_int32", input_shape=(1, 1, 2, 12), dtype=tf.int32)


# ----------------------------------------------------------------------------------------------------------------------
#                                                    Logical operators
# ----------------------------------------------------------------------------------------------------------------------
def gen_unary_logical_operator_test(name, type):
    # Create model.
    inp = layers.Input(name="input1", batch_size=1, shape=2, dtype=tf.bool)
    if type == "not":
        out = tf.math.logical_not(inp)
    else:
        print('Logical unary operator "%s" not supported!')
        exit(1)
    model = Model(inputs=[inp], outputs=[out])
    # Create data.
    inp_tensor = np.array([[False, True]]).astype(bool)
    out_tensor = model.predict([inp_tensor])
    # Save model.
    save_model(model, name)
    # Save data.
    save_tensor(inp_tensor, name + ".inp0")
    save_tensor(out_tensor, name + ".out0")
    # Clear session.
    keras_backend.clear_session()


gen_unary_logical_operator_test(name="logical_not", type="not")


def gen_binary_logical_operator_test(name, type):
    # Create model.
    inp1 = layers.Input(name="input1", batch_size=1, shape=4, dtype=tf.bool)
    inp2 = layers.Input(name="input2", batch_size=1, shape=4, dtype=tf.bool)
    if type == "and":
        out = tf.math.logical_and(inp1, inp2)
    elif type == "or":
        out = tf.math.logical_or(inp1, inp2)
    else:
        print('Logical binary operator "%s" not supported!')
        exit(1)
    model = Model(inputs=[inp1, inp2], outputs=[out])
    # Create data.
    inp1_tensor = np.array([[False, True, False, True]]).astype(bool)
    inp2_tensor = np.array([[False, False, True, True]]).astype(bool)
    out_tensor = model.predict([inp1_tensor, inp2_tensor])
    # Save model.
    save_model(model, name)
    # Save data.
    save_tensor(inp1_tensor, name + ".inp0")
    save_tensor(inp2_tensor, name + ".inp1")
    save_tensor(out_tensor, name + ".out0")
    # Clear session.
    keras_backend.clear_session()


gen_binary_logical_operator_test(name="logical_and", type="and")
gen_binary_logical_operator_test(name="logical_or", type="or")


def gen_cmp_operator_test(name, type):
    # Create model.
    inp1 = layers.Input(name="input1", batch_size=1, shape=3)
    inp2 = layers.Input(name="input2", batch_size=1, shape=3)
    if type == "equal":
        out = tf.math.equal(inp1, inp2)
    elif type == "not_equal":
        out = tf.math.not_equal(inp1, inp2)
    elif type == "less":
        out = tf.math.less(inp1, inp2)
    elif type == "less_equal":
        out = tf.math.less_equal(inp1, inp2)
    elif type == "greater":
        out = tf.math.greater(inp1, inp2)
    elif type == "greater_equal":
        out = tf.math.greater_equal(inp1, inp2)
    else:
        print('Logical operator "%s" not supported!')
        exit(1)
    model = Model(inputs=[inp1, inp2], outputs=[out])
    # Create data.
    inp1_tensor = np.array([[1.0, 1.0, -1.0]]).astype(np.float32)
    inp2_tensor = np.array([[1.0, -1.0, 1.0]]).astype(np.float32)
    out_tensor = model.predict([inp1_tensor, inp2_tensor])
    # Save model.
    save_model(model, name)
    # Save data.
    save_tensor(inp1_tensor, name + ".inp0")
    save_tensor(inp2_tensor, name + ".inp1")
    save_tensor(out_tensor, name + ".out0")
    # Clear session.
    keras_backend.clear_session()


gen_cmp_operator_test(name="equal", type="equal")
gen_cmp_operator_test(name="not_equal", type="not_equal")
gen_cmp_operator_test(name="less", type="less")
gen_cmp_operator_test(name="less_equal", type="less_equal")
gen_cmp_operator_test(name="greater", type="greater")
gen_cmp_operator_test(name="greater_equal", type="greater_equal")


# ----------------------------------------------------------------------------------------------------------------------
#                                                    Unary operators
# ----------------------------------------------------------------------------------------------------------------------
def gen_unary_operator_test(name, type, input_shape):
    # Create model.
    inp = layers.Input(name="input", batch_size=input_shape[0], shape=input_shape[1:])
    if type == "relu":
        out = tf.nn.relu(inp)
    elif type == "relu_n1to1":
        out = tf.clip_by_value(inp, -1.0, 1.0)
    elif type == "relu6":
        out = tf.nn.relu6(inp)
    elif type == "sigmoid":
        out = tf.nn.sigmoid(inp)
    elif type == "exp":
        out = tf.exp(inp)
    elif type == "log":
        out = tf.math.log(inp)
    elif type == "tanh":
        out = tf.nn.tanh(inp)
    elif type == "leaky_relu":
        out = tf.nn.leaky_relu(inp, alpha=0.1)
    elif type == "prelu":
        out = layers.PReLU(alpha_initializer="random_uniform")(inp)
    elif type == "square":
        out = tf.math.square(inp)
    elif type == "abs":
        out = tf.math.abs(inp)
    elif type == "neg":
        out = tf.math.negative(inp)
    elif type == "sqrt":
        out = tf.math.sqrt(inp)
    elif type == "rsqrt":
        out = tf.math.rsqrt(inp)
    elif type == "sin":
        out = tf.math.sin(inp)
    elif type == "cos":
        out = tf.math.cos(inp)
    elif type == "ceil":
        out = tf.math.ceil(inp)
    elif type == "round":
        out = tf.math.round(inp)
    elif type == "floor":
        out = tf.math.floor(inp)
    else:
        print('Unary operator "%s" not supported!')
        exit(1)
    model = Model(inputs=[inp], outputs=[out])
    # Create data.
    np.random.seed(0)
    inp_tensor = np.random.randn(*input_shape).astype(np.float32)
    if type in ["log", "sqrt", "rsqrt"]:
        inp_tensor = np.abs(inp_tensor) + 1
    out_tensor = model.predict(inp_tensor)
    # Save model.
    save_model(model, name)
    # Save data.
    save_tensor(inp_tensor, name + ".inp0")
    save_tensor(out_tensor, name + ".out0")
    # Clear session.
    keras_backend.clear_session()


gen_unary_operator_test(name="relu", type="relu", input_shape=(1, 10))
gen_unary_operator_test(name="relu_n1to1", type="relu_n1to1", input_shape=(1, 10))
gen_unary_operator_test(name="relu6", type="relu6", input_shape=(1, 10))
gen_unary_operator_test(name="sigmoid", type="sigmoid", input_shape=(1, 10))
gen_unary_operator_test(name="tanh", type="tanh", input_shape=(1, 10))
gen_unary_operator_test(name="exp", type="exp", input_shape=(1, 10))
gen_unary_operator_test(name="log", type="log", input_shape=(1, 10))
gen_unary_operator_test(name="leaky_relu", type="leaky_relu", input_shape=(1, 10))
gen_unary_operator_test(name="prelu", type="prelu", input_shape=(1, 10))
gen_unary_operator_test(name="square", type="square", input_shape=(1, 10))
gen_unary_operator_test(name="abs", type="abs", input_shape=(1, 10))
gen_unary_operator_test(name="neg", type="neg", input_shape=(1, 10))
gen_unary_operator_test(name="sqrt", type="sqrt", input_shape=(1, 10))
gen_unary_operator_test(name="rsqrt", type="rsqrt", input_shape=(1, 10))
gen_unary_operator_test(name="sin", type="sin", input_shape=(1, 10))
gen_unary_operator_test(name="cos", type="cos", input_shape=(1, 10))
gen_unary_operator_test(name="ceil", type="ceil", input_shape=(1, 10))
gen_unary_operator_test(name="round", type="round", input_shape=(1, 10))
gen_unary_operator_test(name="floor", type="floor", input_shape=(1, 10))


# ----------------------------------------------------------------------------------------------------------------------
#                                                    Binary operators
# ----------------------------------------------------------------------------------------------------------------------
def gen_binary_operator_test(name, type, input_shape):
    # Create model.
    inp1 = layers.Input(name="input1", batch_size=input_shape[0], shape=input_shape[1:])
    inp2 = layers.Input(name="input2", batch_size=input_shape[0], shape=input_shape[1:])
    if type == "add":
        out = tf.math.add(inp1, inp2)
    elif type == "mul":
        out = tf.math.multiply(inp1, inp2)
    elif type == "sub":
        out = tf.math.subtract(inp1, inp2)
    elif type == "div":
        out = tf.math.divide(inp1, inp2)
    elif type == "pow":
        out = tf.math.pow(inp1, inp2)
    elif type == "max":
        out = tf.math.maximum(inp1, inp2)
    elif type == "min":
        out = tf.math.minimum(inp1, inp2)
    else:
        print('Binary operator "%s" not supported!')
        exit(1)
    model = Model(inputs=[inp1, inp2], outputs=[out])
    # Create data.
    np.random.seed(0)
    inp1_tensor = np.random.rand(*input_shape).astype(np.float32)
    inp2_tensor = np.random.rand(*input_shape).astype(np.float32)
    if type == "pow":
        inp1_tensor = np.abs(inp1_tensor) + 1
    out_tensor = model.predict([inp1_tensor, inp2_tensor])
    # Save model.
    save_model(model, name)
    # Save data.
    save_tensor(inp1_tensor, name + ".inp0")
    save_tensor(inp2_tensor, name + ".inp1")
    save_tensor(out_tensor, name + ".out0")
    # Clear session.
    keras_backend.clear_session()


gen_binary_operator_test(name="add", type="add", input_shape=(1, 10))
gen_binary_operator_test(name="mul", type="mul", input_shape=(1, 10))
gen_binary_operator_test(name="sub", type="sub", input_shape=(1, 10))
gen_binary_operator_test(name="div", type="div", input_shape=(1, 10))
gen_binary_operator_test(name="pow", type="pow", input_shape=(1, 10))
gen_binary_operator_test(name="max", type="max", input_shape=(1, 10))
gen_binary_operator_test(name="min", type="min", input_shape=(1, 10))


# ----------------------------------------------------------------------------------------------------------------------
#                                            Binary broadcasted operators
# ----------------------------------------------------------------------------------------------------------------------
def gen_binary_broadcast_operator_test(name, type, shape_1, shape_2):
    # Create model.
    inp1 = layers.Input(name="input1", batch_size=shape_1[0], shape=shape_1[1:])
    inp2 = layers.Input(name="input2", batch_size=shape_2[0], shape=shape_2[1:])
    if type == "add":
        out = tf.math.add(inp1, inp2)
    elif type == "mul":
        out = tf.math.multiply(inp1, inp2)
    elif type == "sub":
        out = tf.math.subtract(inp1, inp2)
    elif type == "div":
        out = tf.math.divide(inp1, inp2)
    elif type == "max":
        out = tf.math.maximum(inp1, inp2)
    elif type == "min":
        out = tf.math.minimum(inp1, inp2)
    else:
        print('Binary operator "%s" not supported!' % type)
        exit(1)
    model = Model(inputs=[inp1, inp2], outputs=[out])
    # Create data.
    np.random.seed(0)
    inp1_tensor = np.random.rand(*shape_1).astype(np.float32)
    inp2_tensor = np.random.rand(*shape_2).astype(np.float32)
    if type == "pow":
        inp1_tensor = np.abs(inp1_tensor) + 1
    out_tensor = model.predict([inp1_tensor, inp2_tensor])
    # Save model.
    save_model(model, name)
    # Save data.
    save_tensor(inp1_tensor, name + ".inp0")
    save_tensor(inp2_tensor, name + ".inp1")
    save_tensor(out_tensor, name + ".out0")
    # Clear session.
    keras_backend.clear_session()


gen_binary_broadcast_operator_test(
    name="add_broadcast", type="add", shape_1=(1, 5, 5, 3), shape_2=(1, 1, 1, 3)
)
gen_binary_broadcast_operator_test(
    name="mul_broadcast", type="mul", shape_1=(1, 5, 5, 3), shape_2=(1, 1, 1, 3)
)
gen_binary_broadcast_operator_test(
    name="sub_broadcast", type="sub", shape_1=(1, 5, 5, 3), shape_2=(1, 1, 1, 3)
)
gen_binary_broadcast_operator_test(
    name="div_broadcast", type="div", shape_1=(1, 5, 5, 3), shape_2=(1, 1, 1, 3)
)
gen_binary_broadcast_operator_test(
    name="max_broadcast", type="max", shape_1=(1, 5, 5, 3), shape_2=(1, 1, 1, 3)
)
gen_binary_broadcast_operator_test(
    name="min_broadcast", type="min", shape_1=(1, 5, 5, 3), shape_2=(1, 1, 1, 3)
)


# ----------------------------------------------------------------------------------------------------------------------
#                                                        Conv2D
# ----------------------------------------------------------------------------------------------------------------------
def gen_conv2d_test(
    name, input_shape, filters, kernels, strides, padding, dilations, activation
):
    # Create model.
    inp = layers.Input(name="input", batch_size=input_shape[0], shape=input_shape[1:])
    out = layers.Conv2D(
        filters=filters,
        kernel_size=kernels,
        strides=strides,
        padding=padding,
        dilation_rate=dilations,
        activation=activation,
        use_bias=True,
        bias_initializer="random_normal",
    )(inp)
    model = Model(inputs=[inp], outputs=[out])
    # Create data.
    np.random.seed(0)
    inp_tensor = np.random.rand(*input_shape).astype(np.float32)
    out_tensor = model.predict(inp_tensor)
    # Save model.
    save_model(model, name)
    # Save data.
    save_tensor(inp_tensor, name + ".inp0")
    save_tensor(out_tensor, name + ".out0")
    # Clear session.
    keras_backend.clear_session()


gen_conv2d_test(
    name="conv2d_valid",
    input_shape=(1, 5, 5, 3),
    filters=2,
    kernels=(2, 3),
    strides=(1, 1),
    padding="valid",
    dilations=(1, 1),
    activation="linear",
)

gen_conv2d_test(
    name="conv2d_same",
    input_shape=(1, 5, 5, 3),
    filters=2,
    kernels=(2, 3),
    strides=(1, 1),
    padding="same",
    dilations=(1, 1),
    activation="linear",
)

gen_conv2d_test(
    name="conv2d_relu",
    input_shape=(1, 5, 5, 3),
    filters=2,
    kernels=(2, 3),
    strides=(1, 1),
    padding="valid",
    dilations=(1, 1),
    activation="relu",
)


# ----------------------------------------------------------------------------------------------------------------------
#                                                    DepthwiseConv2D
# ----------------------------------------------------------------------------------------------------------------------
def gen_depthwise_conv2d_test(
    name,
    input_shape,
    depth_multiplier,
    kernels,
    strides,
    padding,
    dilations,
    activation,
):
    # Create model.
    inp = layers.Input(name="input", batch_size=input_shape[0], shape=input_shape[1:])
    out = layers.DepthwiseConv2D(
        kernel_size=kernels,
        strides=strides,
        padding=padding,
        depth_multiplier=depth_multiplier,
        dilation_rate=dilations,
        activation=activation,
        use_bias=True,
        bias_initializer="random_normal",
    )(inp)
    model = Model(inputs=[inp], outputs=[out])
    # Create data.
    np.random.seed(0)
    inp_tensor = np.random.rand(*input_shape).astype(np.float32)
    out_tensor = model.predict(inp_tensor)
    # Save model.
    save_model(model, name)
    # Save data.
    save_tensor(inp_tensor, name + ".inp0")
    save_tensor(out_tensor, name + ".out0")
    # Clear session.
    keras_backend.clear_session()


gen_depthwise_conv2d_test(
    name="depthwise_conv2d_c1_m1",
    input_shape=(1, 5, 5, 1),
    depth_multiplier=1,
    kernels=(2, 3),
    strides=(1, 1),
    padding="valid",
    dilations=(1, 1),
    activation="linear",
)

gen_depthwise_conv2d_test(
    name="depthwise_conv2d_c1_m2",
    input_shape=(1, 5, 5, 1),
    depth_multiplier=2,
    kernels=(2, 3),
    strides=(1, 1),
    padding="valid",
    dilations=(1, 1),
    activation="linear",
)

gen_depthwise_conv2d_test(
    name="depthwise_conv2d_c2_m1",
    input_shape=(1, 5, 5, 2),
    depth_multiplier=1,
    kernels=(2, 3),
    strides=(1, 1),
    padding="valid",
    dilations=(1, 1),
    activation="linear",
)

gen_depthwise_conv2d_test(
    name="depthwise_conv2d_c2_m2",
    input_shape=(1, 5, 5, 2),
    depth_multiplier=2,
    kernels=(2, 3),
    strides=(1, 1),
    padding="valid",
    dilations=(1, 1),
    activation="linear",
)


# ----------------------------------------------------------------------------------------------------------------------
#                                                        FullyConnected
# ----------------------------------------------------------------------------------------------------------------------
def gen_fully_connected_test(name, input_shape, out_channels, activation):
    # Create model.
    inp = layers.Input(name="input", batch_size=input_shape[0], shape=input_shape[1:])
    out = layers.Dense(
        units=out_channels,
        activation=activation,
        use_bias=True,
        bias_initializer="random_normal",
    )(inp)
    model = Model(inputs=[inp], outputs=[out])
    # Create data.
    np.random.seed(0)
    inp_tensor = np.random.rand(*input_shape).astype(np.float32)
    out_tensor = model.predict(inp_tensor)
    # Save model.
    save_model(model, name)
    # Save data.
    save_tensor(inp_tensor, name + ".inp0")
    save_tensor(out_tensor, name + ".out0")
    # Clear session.
    keras_backend.clear_session()


gen_fully_connected_test(
    name="fully_connected", input_shape=(2, 5), out_channels=10, activation="linear"
)


# ----------------------------------------------------------------------------------------------------------------------
#                                                        MaxPool2D
# ----------------------------------------------------------------------------------------------------------------------
def gen_maxpool2d_test(name, input_shape, kernels, strides, padding):
    # Create model.
    inp = layers.Input(name="input", batch_size=input_shape[0], shape=input_shape[1:])
    out = layers.MaxPooling2D(pool_size=kernels, strides=strides, padding=padding)(inp)
    model = Model(inputs=[inp], outputs=[out])
    # Create data.
    np.random.seed(0)
    inp_tensor = np.random.rand(*input_shape).astype(np.float32)
    out_tensor = model.predict(inp_tensor)
    # Save model.
    save_model(model, name)
    # Save data.
    save_tensor(inp_tensor, name + ".inp0")
    save_tensor(out_tensor, name + ".out0")
    # Clear session.
    keras_backend.clear_session()


gen_maxpool2d_test(
    name="maxpool2d_valid",
    input_shape=(1, 5, 5, 2),
    kernels=(2, 3),
    strides=(1, 1),
    padding="valid",
)

gen_maxpool2d_test(
    name="maxpool2d_same",
    input_shape=(1, 5, 5, 2),
    kernels=(2, 3),
    strides=(1, 1),
    padding="same",
)


# ----------------------------------------------------------------------------------------------------------------------
#                                                        AvgPool2D
# ----------------------------------------------------------------------------------------------------------------------
def gen_avgpool2d_test(name, input_shape, kernels, strides, padding):
    # Create model.
    inp = layers.Input(name="input", batch_size=input_shape[0], shape=input_shape[1:])
    out = layers.AveragePooling2D(pool_size=kernels, strides=strides, padding=padding)(
        inp
    )
    model = Model(inputs=[inp], outputs=[out])
    # Create data.
    np.random.seed(0)
    inp_tensor = np.random.rand(*input_shape).astype(np.float32)
    out_tensor = model.predict(inp_tensor)
    # Save model.
    save_model(model, name)
    # Save data.
    save_tensor(inp_tensor, name + ".inp0")
    save_tensor(out_tensor, name + ".out0")
    # Clear session.
    keras_backend.clear_session()


gen_avgpool2d_test(
    name="avgpool2d_valid",
    input_shape=(1, 5, 5, 2),
    kernels=(2, 3),
    strides=(1, 1),
    padding="valid",
)

gen_avgpool2d_test(
    name="avgpool2d_same",
    input_shape=(1, 5, 5, 2),
    kernels=(2, 3),
    strides=(1, 1),
    padding="same",
)


# ----------------------------------------------------------------------------------------------------------------------
#                                                        Softmax
# ----------------------------------------------------------------------------------------------------------------------
def gen_softmax_test(name, input_shape, axis):
    # Create model.
    inp = layers.Input(name="input", batch_size=input_shape[0], shape=input_shape[1:])
    out = layers.Softmax(axis=axis)(inp)
    model = Model(inputs=[inp], outputs=[out])
    # Create data.
    np.random.seed(0)
    inp_tensor = np.random.rand(*input_shape).astype(np.float32)
    out_tensor = model.predict(inp_tensor)
    # Save model.
    save_model(model, name)
    # Save data.
    save_tensor(inp_tensor, name + ".inp0")
    save_tensor(out_tensor, name + ".out0")
    # Clear session.
    keras_backend.clear_session()


gen_softmax_test(name="softmax", input_shape=(1, 3), axis=-1)


# ----------------------------------------------------------------------------------------------------------------------
#                                                        Transpose
# ----------------------------------------------------------------------------------------------------------------------
def gen_transpose_test(name, input_shape, perm):
    # Create model.
    inp = layers.Input(name="input", batch_size=input_shape[0], shape=input_shape[1:])
    out = layers.Permute(perm)(inp)
    model = Model(inputs=[inp], outputs=[out])
    # Create data.
    np.random.seed(0)
    inp_tensor = np.random.rand(*input_shape).astype(np.float32)
    out_tensor = model.predict(inp_tensor)
    # Save model.
    save_model(model, name)
    # Save data.
    save_tensor(inp_tensor, name + ".inp0")
    save_tensor(out_tensor, name + ".out0")
    # Clear session.
    keras_backend.clear_session()


gen_transpose_test(name="transpose", input_shape=(1, 2, 3), perm=(2, 1))


# ----------------------------------------------------------------------------------------------------------------------
#                                                        Slice
# ----------------------------------------------------------------------------------------------------------------------
def gen_slice_test(name, input_shape, begin, size):
    # Create model.
    inp = layers.Input(name="input", batch_size=input_shape[0], shape=input_shape[1:])
    out = tf.slice(inp, begin, size)
    model = Model(inputs=[inp], outputs=[out])
    # Create data.
    np.random.seed(0)
    inp_tensor = np.random.rand(*input_shape).astype(np.float32)
    out_tensor = model.predict(inp_tensor)
    # Save model.
    save_model(model, name)
    # Save data.
    save_tensor(inp_tensor, name + ".inp0")
    save_tensor(out_tensor, name + ".out0")
    # Clear session.
    keras_backend.clear_session()


gen_slice_test(name="slice", input_shape=(1, 2, 3), begin=(0, 1, 2), size=(1, 1, 1))

gen_slice_test(
    name="slice_neg_size", input_shape=(1, 2, 3), begin=(0, 1, 2), size=(1, 1, -1)
)


# ----------------------------------------------------------------------------------------------------------------------
#                                                        Reshape
# ----------------------------------------------------------------------------------------------------------------------
def gen_reshape_test(name, input_shape, shape):
    # Create model.
    inp = layers.Input(name="input", batch_size=input_shape[0], shape=input_shape[1:])
    out = tf.reshape(inp, shape)
    model = Model(inputs=[inp], outputs=[out])
    # Create data.
    np.random.seed(0)
    inp_tensor = np.random.rand(*input_shape).astype(np.float32)
    out_tensor = model.predict(inp_tensor)
    # Save model.
    save_model(model, name)
    # Save data.
    save_tensor(inp_tensor, name + ".inp0")
    save_tensor(out_tensor, name + ".out0")
    # Clear session.
    keras_backend.clear_session()


gen_reshape_test(name="reshape", input_shape=(1, 2, 3), shape=(1, 6))

gen_reshape_test(name="reshape_neg_shape", input_shape=(1, 2, 3), shape=(1, -1))


# ----------------------------------------------------------------------------------------------------------------------
#                                                        Concat
# ----------------------------------------------------------------------------------------------------------------------
def gen_concat_test(name, input_shape, axis):
    # Create model.
    inp1 = layers.Input(name="input1", batch_size=input_shape[0], shape=input_shape[1:])
    inp2 = layers.Input(name="input2", batch_size=input_shape[0], shape=input_shape[1:])
    out = tf.concat([inp1, inp2], axis)
    model = Model(inputs=[inp1, inp2], outputs=[out])
    # Create data.
    np.random.seed(0)
    inp1_tensor = np.random.rand(*input_shape).astype(np.float32)
    inp2_tensor = np.random.rand(*input_shape).astype(np.float32)
    out_tensor = model.predict([inp1_tensor, inp2_tensor])
    # Save model.
    save_model(model, name)
    # Save data.
    save_tensor(inp1_tensor, name + ".inp0")
    save_tensor(inp2_tensor, name + ".inp1")
    save_tensor(out_tensor, name + ".out0")
    # Clear session.
    keras_backend.clear_session()


gen_concat_test(name="concat", input_shape=(1, 2, 3), axis=1)

gen_concat_test(name="concat_neg_axis", input_shape=(1, 2, 3), axis=-1)


# ----------------------------------------------------------------------------------------------------------------------
#                                                    Split
# ----------------------------------------------------------------------------------------------------------------------
def gen_split_test(name, input_shape, axis, num_split):
    # Create model.
    inp = layers.Input(name="input", batch_size=input_shape[0], shape=input_shape[1:])
    outs = tf.split(inp, num_or_size_splits=num_split, axis=axis)
    model = Model(inputs=[inp], outputs=outs)
    # Create data.
    np.random.seed(0)
    inp_tensor = np.random.rand(*input_shape).astype(np.float32)
    out_tensors = model.predict(inp_tensor)
    # Save model.
    save_model(model, name)
    # Save data.
    save_tensor(inp_tensor, name + ".inp0")
    for idx in range(len(out_tensors)):
        save_tensor(out_tensors[idx], name + (".out%d" % idx))
    # Clear session.
    keras_backend.clear_session()


gen_split_test(name="split", input_shape=(1, 9), axis=-1, num_split=3)


# ----------------------------------------------------------------------------------------------------------------------
#                                                        Pad
# ----------------------------------------------------------------------------------------------------------------------
def gen_pad_test(name, input_shape, pads):
    # Create model.
    inp = layers.Input(name="input", batch_size=input_shape[0], shape=input_shape[1:])
    out = tf.pad(inp, paddings=pads)
    model = Model(inputs=[inp], outputs=[out])
    # Create data.
    np.random.seed(0)
    inp_tensor = np.random.rand(*input_shape).astype(np.float32)
    out_tensor = model.predict(inp_tensor)
    # Save model.
    save_model(model, name)
    # Save data.
    save_tensor(inp_tensor, name + ".inp0")
    save_tensor(out_tensor, name + ".out0")
    # Clear session.
    keras_backend.clear_session()


gen_pad_test(name="pad", input_shape=(1, 2, 2), pads=[[0, 0], [1, 2], [0, 3]])


# ----------------------------------------------------------------------------------------------------------------------
#                                                        ArgMin/ArgMax
# ----------------------------------------------------------------------------------------------------------------------
def gen_arg_min_max_test(name, type, input_shape, axis):
    # Create model.
    inp = layers.Input(name="input", batch_size=input_shape[0], shape=input_shape[1:])
    if type == "min":
        out = tf.math.argmin(inp, axis=axis)
    else:
        out = tf.math.argmax(inp, axis=axis)
    model = Model(inputs=[inp], outputs=[out])
    # Create data.
    np.random.seed(0)
    inp_tensor = np.random.rand(*input_shape).astype(np.float32)
    out_tensor = model.predict(inp_tensor)
    # Save model.
    save_model(model, name)
    # Save data.
    save_tensor(inp_tensor, name + ".inp0")
    save_tensor(out_tensor, name + ".out0")
    # Clear session.
    keras_backend.clear_session()


gen_arg_min_max_test(name="arg_min", type="min", input_shape=(1, 2, 10), axis=2)
gen_arg_min_max_test(name="arg_max", type="max", input_shape=(1, 2, 10), axis=2)


# ----------------------------------------------------------------------------------------------------------------------
#                                                        Pack
# ----------------------------------------------------------------------------------------------------------------------
def gen_pack_test(name, input_shape, axis):
    # Create model.
    inp1 = layers.Input(name="input1", batch_size=input_shape[0], shape=input_shape[1:])
    inp2 = layers.Input(name="input2", batch_size=input_shape[0], shape=input_shape[1:])
    out = tf.stack([inp1, inp2], axis=axis)
    model = Model(inputs=[inp1, inp2], outputs=[out])
    # Create data.
    np.random.seed(0)
    inp1_tensor = np.random.rand(*input_shape).astype(np.float32)
    inp2_tensor = np.random.rand(*input_shape).astype(np.float32)
    out_tensor = model.predict([inp1_tensor, inp2_tensor])
    # Save model.
    save_model(model, name)
    # Save data.
    save_tensor(inp1_tensor, name + ".inp0")
    save_tensor(inp2_tensor, name + ".inp1")
    save_tensor(out_tensor, name + ".out0")
    # Clear session.
    keras_backend.clear_session()


gen_pack_test(name="pack", input_shape=(2, 3, 4), axis=1)


# ----------------------------------------------------------------------------------------------------------------------
#                                                       Unpack
# ----------------------------------------------------------------------------------------------------------------------
def gen_unpack_test(name, input_shape, axis):
    # Create model.
    inp = layers.Input(name="input", batch_size=input_shape[0], shape=input_shape[1:])
    outs = tf.unstack(inp, axis=axis)
    model = Model(inputs=[inp], outputs=outs)
    # Create data.
    np.random.seed(0)
    inp_tensor = np.random.rand(*input_shape).astype(np.float32)
    out_tensors = model.predict(inp_tensor)
    # Save model.
    save_model(model, name)
    # Save data.
    save_tensor(inp_tensor, name + ".inp0")
    for idx in range(len(out_tensors)):
        save_tensor(out_tensors[idx], name + (".out%d" % idx))
    # Clear session.
    keras_backend.clear_session()


gen_unpack_test(name="unpack", input_shape=(2, 3, 4), axis=1)


# ----------------------------------------------------------------------------------------------------------------------
#                                                        Mean
# ----------------------------------------------------------------------------------------------------------------------
def gen_mean_test(name, input_shape, axis, keep_dims):
    # Create model.
    inp = layers.Input(name="input", batch_size=input_shape[0], shape=input_shape[1:])
    out = tf.reduce_mean(inp, axis=axis, keepdims=keep_dims)
    model = Model(inputs=[inp], outputs=[out])
    # Create data.
    np.random.seed(0)
    inp_tensor = np.random.rand(*input_shape).astype(np.float32)
    out_tensor = model.predict(inp_tensor)
    # Save model.
    save_model(model, name)
    # Save data.
    save_tensor(inp_tensor, name + ".inp0")
    save_tensor(out_tensor, name + ".out0")
    # Clear session.
    keras_backend.clear_session()


gen_mean_test(name="mean_keep_dims", input_shape=(1, 2, 10), axis=2, keep_dims=True)
gen_mean_test(name="mean_no_keep_dims", input_shape=(1, 2, 10), axis=2, keep_dims=False)
gen_mean_test(
    name="mean_multiple_axis_keep_dims",
    input_shape=(1, 2, 10),
    axis=(1, 2),
    keep_dims=True,
)
gen_mean_test(
    name="mean_multiple_axis_no_keep_dims",
    input_shape=(1, 2, 10),
    axis=(1, 2),
    keep_dims=False,
)


# ----------------------------------------------------------------------------------------------------------------------
#                                                        Tile
# ----------------------------------------------------------------------------------------------------------------------
def gen_tile_test(name, input_shape, tiles):
    # Create model.
    inp = layers.Input(name="input", batch_size=input_shape[0], shape=input_shape[1:])
    out = tf.tile(inp, multiples=tiles)
    model = Model(inputs=[inp], outputs=[out])
    # Create data.
    np.random.seed(0)
    inp_tensor = np.random.rand(*input_shape).astype(np.float32)
    out_tensor = model.predict(inp_tensor)
    # Save model.
    save_model(model, name)
    # Save data.
    save_tensor(inp_tensor, name + ".inp0")
    save_tensor(out_tensor, name + ".out0")
    # Clear session.
    keras_backend.clear_session()


gen_tile_test(name="tile", input_shape=(1, 2, 3), tiles=[1, 3, 2])


# ----------------------------------------------------------------------------------------------------------------------
#                                                     RESIZE NEAREST
# ----------------------------------------------------------------------------------------------------------------------
def gen_resize_nearest_test(name, input_shape, output_shape):
    # Create model.
    inp = layers.Input(name="input", batch_size=input_shape[0], shape=input_shape[1:])
    out = tf.compat.v1.image.resize_nearest_neighbor(inp, size=output_shape)
    model = Model(inputs=[inp], outputs=[out])
    # Create data.
    np.random.seed(0)
    inp_tensor = np.random.rand(*input_shape).astype(np.float32)
    out_tensor = model.predict(inp_tensor)
    # Save model.
    save_model(model, name)
    # Save data.
    save_tensor(inp_tensor, name + ".inp0")
    save_tensor(out_tensor, name + ".out0")
    # Clear session.
    keras_backend.clear_session()


gen_resize_nearest_test(
    name="resize_nearest", input_shape=(1, 3, 4, 2), output_shape=(5, 7)
)


# ----------------------------------------------------------------------------------------------------------------------
#                                                     RESIZE BILINEAR
# ----------------------------------------------------------------------------------------------------------------------
def gen_resize_bilinear_test(name, input_shape, output_shape):
    # Create model.
    inp = layers.Input(name="input", batch_size=input_shape[0], shape=input_shape[1:])
    out = tf.compat.v1.image.resize_bilinear(inp, size=output_shape)
    model = Model(inputs=[inp], outputs=[out])
    # Create data.
    np.random.seed(0)
    inp_tensor = np.random.rand(*input_shape).astype(np.float32)
    out_tensor = model.predict(inp_tensor)
    # Save model.
    save_model(model, name)
    # Save data.
    save_tensor(inp_tensor, name + ".inp0")
    save_tensor(out_tensor, name + ".out0")
    # Clear session.
    keras_backend.clear_session()


gen_resize_bilinear_test(
    name="resize_bilinear", input_shape=(1, 3, 4, 2), output_shape=(5, 7)
)


# ----------------------------------------------------------------------------------------------------------------------
#                                                    SPACE TO DEPTH
# ----------------------------------------------------------------------------------------------------------------------
def gen_space_to_depth_test(name, input_shape, block_size):
    # Create model.
    inp = layers.Input(name="input", batch_size=input_shape[0], shape=input_shape[1:])
    out = tf.compat.v1.space_to_depth(inp, block_size=block_size)
    model = Model(inputs=[inp], outputs=[out])
    # Create data.
    np.random.seed(0)
    inp_tensor = np.random.rand(*input_shape).astype(np.float32)
    out_tensor = model.predict(inp_tensor)
    # Save model.
    save_model(model, name)
    # Save data.
    save_tensor(inp_tensor, name + ".inp0")
    save_tensor(out_tensor, name + ".out0")
    # Clear session.
    keras_backend.clear_session()


gen_space_to_depth_test(name="space_to_depth", input_shape=(1, 2, 4, 3), block_size=2)


# ----------------------------------------------------------------------------------------------------------------------
#                                                     DEPTH TO SPACE
# ----------------------------------------------------------------------------------------------------------------------
# Note: Older version of TensorFlow handles this operator as custom. This test is generated separately by manually
# editing the 'space_to_depth' test.
def gen_depth_to_space_test(name, input_shape, block_size):
    # Create model.
    inp = layers.Input(name="input", batch_size=input_shape[0], shape=input_shape[1:])
    out = tf.nn.depth_to_space(inp, block_size=block_size)
    model = Model(inputs=[inp], outputs=[out])
    # Create data.
    np.random.seed(0)
    inp_tensor = np.random.rand(*input_shape).astype(np.float32)
    out_tensor = model.predict(inp_tensor)
    # Save model.
    save_model(model, name)
    # Save data.
    save_tensor(inp_tensor, name + ".inp0")
    save_tensor(out_tensor, name + ".out0")
    # Clear session.
    keras_backend.clear_session()


gen_depth_to_space_test(name="depth_to_space", input_shape=(1, 1, 2, 12), block_size=2)
