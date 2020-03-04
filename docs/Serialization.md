# Serialization in Glow

Glow supports multiple pathways for both exporting and loading a serialized model.

## Loading

Glow can load Caffe2 and ONNX protobufs, as well as TorchScript exported from a
PyTorch model. For ONNX protobufs, Glow supports loading ops specified by the
[ONNX spec](https://github.com/onnx/onnx/blob/master/docs/Operators.md), as well
as custom ops that are only used by Glow (see below).

## Exporting

Glow supports exporting ONNX protobufs. One way this is done is by trying to
abide by the ONNX spec for ops which can be represented in ONNX. However this
can be problematic as not all ops may fit into ONNX. Additionally, if the
intention is to later on reload this serialized model into Glow, then there is
downside in moving to ONNX and back, as some information may be lost in the
conversion from Glow -> ONNX -> Glow.

As an alternative, Glow also supports exporting to custom ONNX ops, which are
annotated with attributes to ensure that the Glow graph that is exported is the
exact same as the one that is reloaded. See the `useGlowCustomOps_` flag in the
`ONNXModelWriter`.

### Format of custom Glow ops in ONNX

Glow has unique names used for all `Nodes` in a Function, and can specify unique
names for each `NodeValue` based on its result number. Therefore it's natural to
use these to specify input and output names in protobuf form. This is done via
calls to `NodeValue::generateNodeOutputName()`.

Additionally, it's desirable for an exported and reimported graph to retain the
same names for all `Nodes` and `Storage` (`Placeholders` and
`Constants`). Keeping the same name for inputs, nodes, and initializers in ONNX
is easily done by setting the name of the corresponding proto object. However
for outputs, we need to retain both the name of the `SaveNode` as well as the
name of the `Placeholder`. We use the `doc_string` of the output to retain the
name of the `Placeholder`, and use the name of the output itself to retain the
name of the `SaveNode`.

Another note is that we use ONNX `Identity` ops to connect an output name of a
`Node` to a `SaveNode` if it is indeed used by a `SaveNode`. This allows us to
simplify the export process, i.e. we do not need to check the users of a `Node`
and decide whether to overwrite its name to match the `SaveNode` name.

In order to retain properties of `Storage`, the `doc_string` of inputs and
outputs is used to save these properties. This currently includes:

* `offline`: Used for static Placeholders
* `layout`: Used for the layout of Storage
* `trainable`: Used for the trainable property of Placeholders
* `ElemKind`: Save the ElemKind of Storage
* `qScale`: Scale of quantized ElemKinds
* `qOffset`: Offset of quantized ElemKinds
* `saveName`: Used for retaining the name of SaveNodes used by output Placeholders

You can find a couple examples of some tests that were exported with the
`useGlowCustomOps` flag mentioned above here:
[Quantized TopK](https://github.com/pytorch/glow/tree/master/tests/models/onnxModels/glow_custom_op_topk_quantized.onnxtxt)
and
[ChannelwiseQuantizedConvolution](https://github.com/pytorch/glow/tree/master/tests/models/onnxModels/glow_custom_op_channelwise_quantized_group_conv.onnxtxt).
