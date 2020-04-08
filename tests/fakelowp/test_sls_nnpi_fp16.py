from __future__ import absolute_import, division, print_function, unicode_literals

import time

# Must happen before importing caffe2.python.*
import glow.fb.test.init_shared_libs  # noqa
import numpy as np
from caffe2.proto import caffe2_pb2
from caffe2.python import core, workspace
from caffe2.python.onnx.onnxifi import onnxifi_caffe2_net
from caffe2.python.onnx.tests.test_utils import TestCase
from glow.fb.test.test_utils import print_test_debug_info


workspace.GlobalInit(
    [
        "caffe2",
        "--glow_global_fp16=1",
        "--glow_global_fused_scale_offset_fp16=1",
        "--glow_global_force_sls_fp16_accum=1",
    ]
)
GLOW_MATMUL_ATOL = 1e-5
GLOW_MATMUL_RTOL = 1e-3


class SparseLengthsSumTest(TestCase):
    def Test_SLS_NonQuantized_fp16(self):
        N = 20000
        DIM = 64
        D = (4 * np.random.random_sample((N, DIM)) + 1).astype(np.float32)
        I = (np.random.randint(0, N, size=12)).astype(np.int64)
        L = np.asarray([4, 4, 4]).astype(np.int32)
        workspace.FeedBlob("D", D)

        ref_c2_net = core.Net("test_ref_c2")
        ref_c2_net.SparseLengthsSum(["D", "I", "L"], "ref_out")
        ref_c2_net.Proto().external_input.extend(["D", "I", "L"])
        ref_c2_net.Proto().external_output.extend(["ref_out"])

        fp16_c2_net = core.Net("test_fp16_c2")
        fp16_c2_net.SparseLengthsSumFakeFP16AccFP16(["D", "I", "L"], "fp16_out")

        input_dict = {}

        pred_net = caffe2_pb2.NetDef()
        pred_net.name = "pred"
        pred_net.external_input.extend(["D", "I", "L"])
        pred_net.external_output.append("glow_out")
        pred_net.op.add().CopyFrom(
            core.CreateOperator("SparseLengthsSum", ["D", "I", "L"], ["glow_out"])
        )

        onnxified_net = onnxifi_caffe2_net(
            pred_net,
            input_dict,
            max_batch_size=3,
            max_seq_size=16,
            debug=True,
            adjust_batch=False,
            use_onnx=False,
        )

        num_onnxified_ops = sum(
            1 if op.type == "Onnxifi" else 0 for op in onnxified_net.op
        )
        print(onnxified_net)
        np.testing.assert_equal(num_onnxified_ops, 1)

        workspace.FeedBlob("I", I)
        workspace.FeedBlob("L", L)

        workspace.RunNetOnce(ref_c2_net)
        ref_c2_out = workspace.FetchBlob("ref_out")

        workspace.RunNetOnce(fp16_c2_net)
        fp16_c2_out = workspace.FetchBlob("fp16_out")

        np.testing.assert_allclose(fp16_c2_out, ref_c2_out, atol=1e-3, rtol=1e-3)

        workspace.RunNetOnce(onnxified_net)
        fp16_glow_out = workspace.FetchBlob("glow_out")

        if not np.allclose(fp16_glow_out, fp16_c2_out):
            diff = np.abs(fp16_glow_out - fp16_c2_out)
            print_test_debug_info(
                "sls",
                {
                    "indices": I,
                    "data": D,
                    "lengths": L,
                    "Y_c2": fp16_c2_out,
                    "Y_glow": fp16_glow_out,
                    "diff": diff,
                    "rowwise_diff": diff[:, 0],
                },
            )
            assert 0

    def test_slws_fused_8bit_rowwise_all_same(self):
        # Comment out for predictable debugging
        np.random.seed(int(time.time()))
        workspace.ResetWorkspace()
        n = 1
        m = 2
        data = np.ones((n, m)).astype(np.float32) * 0.2 - 0.1

        max_segments = 5
        max_segment_length = 200
        num_lengths = np.random.randint(1, max_segments + 1)
        # number of segments to run
        lengths = np.random.randint(0, max_segment_length + 1, size=num_lengths).astype(
            np.int32
        )
        num_indices = np.sum(lengths)
        indices = np.zeros(num_indices, dtype=np.int64)
        weights = np.random.uniform(low=-0.5, high=0.5, size=[len(indices)]).astype(
            np.float32
        )

        pred_net = caffe2_pb2.NetDef()
        pred_net.name = "pred"
        pred_net.external_input.extend(
            ["quantized_data", "weights", "indices", "lengths"]
        )
        pred_net.external_output.append("Y")
        pred_net.op.add().CopyFrom(
            core.CreateOperator(
                "SparseLengthsWeightedSumFused8BitRowwise",
                ["quantized_data", "weights", "indices", "lengths"],
                ["Y"],
            )
        )

        ref_net = caffe2_pb2.NetDef()
        ref_net.name = "ref"
        ref_net.external_input.extend(
            ["quantized_data", "weights", "indices", "lengths"]
        )
        ref_net.external_output.append("Y")
        ref_net.op.add().CopyFrom(
            core.CreateOperator(
                "SparseLengthsWeightedSumFused8BitRowwiseFakeFP16NNPI",
                ["quantized_data", "weights", "indices", "lengths"],
                ["Y"],
            )
        )

        workspace.FeedBlob("data", data)
        workspace.RunOperatorOnce(
            core.CreateOperator(
                "FloatToFused8BitRowwiseQuantized", ["data"], ["quantized_data"]
            )
        )
        pred_net_onnxified = onnxifi_caffe2_net(
            pred_net,
            {},
            max_batch_size=max_segments,
            max_seq_size=max_segments * max_segment_length,
            debug=True,
            adjust_batch=True,
            use_onnx=False,
        )

        num_onnxified_ops = sum(
            1 if o.type == "Onnxifi" else 0 for o in pred_net_onnxified.op
        )
        np.testing.assert_equal(num_onnxified_ops, 1)

        workspace.FeedBlob("indices", indices)
        workspace.FeedBlob("lengths", lengths)
        workspace.FeedBlob("weights", weights)

        workspace.CreateNet(pred_net_onnxified)
        workspace.CreateNet(ref_net)

        workspace.RunNet(pred_net_onnxified.name)
        Y_glow = workspace.FetchBlob("Y")

        workspace.RunNet(ref_net.name)
        Y_c2 = workspace.FetchBlob("Y")

        if not np.allclose(Y_c2, Y_glow):
            print_test_debug_info(
                "slws_fused_8bit_rowwise",
                {
                    "indices": indices,
                    "data": data,
                    "lengths": lengths,
                    "weights": weights,
                    "Y_c2": Y_c2,
                    "Y_glow": Y_glow,
                    "diff": Y_glow - Y_c2,
                    "rowwise_diff": (Y_glow - Y_c2)[:, 0],
                },
            )
            assert 0

    def test_slws_fused_8bit_rowwise_turkey(self):
        # Comment out for predictable debugging
        seed = int(time.time() * 1000) % 2 ** 16
        print(seed)
        np.random.seed(seed)
        workspace.ResetWorkspace()

        n = 20000
        DIM = 6
        data = (4 * np.random.random_sample((n, DIM)) + 1).astype(np.float32)

        max_segments = 200
        max_segment_length = 200
        num_lengths = np.random.randint(0, max_segments + 1)
        # number of segments to run
        lengths = np.random.randint(2, max_segment_length + 1, size=num_lengths).astype(
            np.int32
        )
        num_indices = np.sum(lengths)
        indices = np.random.randint(low=0, high=n, size=num_indices, dtype=np.int64)
        weights = np.random.uniform(low=0.01, high=0.5, size=[len(indices)]).astype(
            np.float32
        )

        pred_net = caffe2_pb2.NetDef()
        pred_net.name = "pred"
        pred_net.external_input.extend(
            ["quantized_data", "weights", "indices", "lengths"]
        )
        pred_net.external_output.append("Y")
        pred_net.op.add().CopyFrom(
            core.CreateOperator(
                "SparseLengthsWeightedSumFused8BitRowwise",
                ["quantized_data", "weights", "indices", "lengths"],
                ["Y"],
            )
        )

        ref_net = caffe2_pb2.NetDef()
        ref_net.name = "ref"
        ref_net.external_input.extend(
            ["quantized_data", "weights", "indices", "lengths"]
        )
        ref_net.external_output.append("Y")
        ref_net.op.add().CopyFrom(
            core.CreateOperator(
                "SparseLengthsWeightedSumFused8BitRowwiseFakeFP16NNPI",
                ["quantized_data", "weights", "indices", "lengths"],
                ["Y"],
            )
        )

        workspace.FeedBlob("data", data)
        workspace.RunOperatorOnce(
            core.CreateOperator(
                "FloatToFused8BitRowwiseQuantized", ["data"], ["quantized_data"]
            )
        )
        onnxified_net = onnxifi_caffe2_net(
            pred_net,
            {},
            max_batch_size=max_segments,
            max_seq_size=max_segments * max_segment_length,
            debug=True,
            adjust_batch=True,
            use_onnx=False,
        )
        workspace.FeedBlob("indices", indices)
        workspace.FeedBlob("lengths", lengths)
        workspace.FeedBlob("weights", weights)

        workspace.CreateNet(onnxified_net)
        workspace.CreateNet(ref_net)

        workspace.RunNet(onnxified_net.name)
        Y_glow = workspace.FetchBlob("Y")

        workspace.RunNet(ref_net.name)
        Y_ref = workspace.FetchBlob("Y")

        diff = np.abs((Y_ref - Y_glow) / (Y_ref + 1e-8))
        max_err = np.max(diff, axis=1)
        num_offenders = (max_err > 0).sum()
        if num_offenders > 0:
            print_test_debug_info(
                "slws_fused_8bit_rowwise_inv_scale",
                {
                    "indices": indices,
                    "data": data.shape,
                    "lengths": lengths,
                    "weights": weights,
                    "Y_glow": Y_glow,
                    "Y_ref": Y_ref,
                    "diff": diff,
                    "rowwise_diff": np.max(diff, axis=1),
                },
            )
            assert 0

    # Simple test to aid debugging order of operations
    # Minimize the case to an SLS that adds two rows
    def test_small_sls(self):
        seed = int(time.time() * 1000) % 2 ** 16
        print(seed)
        np.random.seed(seed)
        workspace.ResetWorkspace()

        n = 2
        DIM = 3
        data = 4 * (np.random.random_sample((n, DIM)) + 1).astype(np.float32)

        lengths = np.array([n], dtype=np.int32)
        indices = np.array(range(n), dtype=np.int64)
        weights = np.random.uniform(low=0.01, high=0.5, size=[n]).astype(np.float32)

        pred_net = caffe2_pb2.NetDef()
        pred_net.name = "pred"
        pred_net.external_input.extend(
            ["quantized_data", "weights", "indices", "lengths"]
        )
        pred_net.external_output.append("Y")
        pred_net.op.add().CopyFrom(
            core.CreateOperator(
                "SparseLengthsWeightedSumFused8BitRowwise",
                ["quantized_data", "weights", "indices", "lengths"],
                ["Y"],
            )
        )

        ref_net = caffe2_pb2.NetDef()
        ref_net.name = "ref"
        ref_net.external_input.extend(
            ["quantized_data", "weights", "indices", "lengths"]
        )
        ref_net.external_output.append("Y")
        ref_net.op.add().CopyFrom(
            core.CreateOperator(
                "SparseLengthsWeightedSumFused8BitRowwiseFakeFP16NNPI",
                ["quantized_data", "weights", "indices", "lengths"],
                ["Y"],
            )
        )

        workspace.FeedBlob("data", data)
        workspace.RunOperatorOnce(
            core.CreateOperator(
                "FloatToFused8BitRowwiseQuantized", ["data"], ["quantized_data"]
            )
        )

        quantized_data = workspace.FetchBlob("quantized_data")

        onnxified_net = onnxifi_caffe2_net(
            pred_net,
            {},
            max_batch_size=1,
            max_seq_size=n,
            debug=True,
            adjust_batch=True,
            use_onnx=False,
        )
        workspace.FeedBlob("indices", indices)
        workspace.FeedBlob("lengths", lengths)
        workspace.FeedBlob("weights", weights)

        workspace.CreateNet(onnxified_net)
        workspace.CreateNet(ref_net)

        workspace.RunNet(onnxified_net.name)
        Y_glow = workspace.FetchBlob("Y")

        workspace.RunNet(ref_net.name)
        Y_ref = workspace.FetchBlob("Y")

        diff = np.abs((Y_ref - Y_glow) / (Y_ref + 1e-8))
        max_err = np.max(diff, axis=1)
        num_offenders = (max_err > 0).sum()
        if num_offenders > 0:
            np.set_printoptions(precision=12)
            print(
                "ref",
                Y_ref.astype(np.float16).astype(np.float32),
                "glow",
                Y_glow.astype(np.float16).astype(np.float32),
            )
            print_test_debug_info(
                "slws_fused_8bit_rowwise_inv_scale",
                {
                    "seed": seed,
                    "indices": indices,
                    "data": data,
                    "quantized_data": quantized_data,
                    "lengths": lengths,
                    "weights": weights,
                    "Y_glow": Y_glow,
                    "Y_ref": Y_ref,
                    "diff": diff,
                    "rowwise_diff": np.max(diff, axis=1),
                },
            )
            assert 0

    def test_small_sls_acc32(self):

        workspace.GlobalInit(
            [
                "caffe2",
                "--glow_global_fp16=0",
                "--glow_global_fused_scale_offset_fp16=0",
                "--glow_global_force_sls_fp16_accum=0",
            ]
        )
        seed = int(time.time() * 1000) % 2 ** 16
        print(seed)
        np.random.seed(seed)
        workspace.ResetWorkspace()

        n = 2
        DIM = 3
        data = 4 * (np.random.random_sample((n, DIM)) + 1).astype(np.float32)

        lengths = np.array([n], dtype=np.int32)
        indices = np.array(range(n), dtype=np.int64)
        weights = np.random.uniform(low=0.01, high=0.5, size=[n]).astype(np.float32)

        pred_net = caffe2_pb2.NetDef()
        pred_net.name = "pred"
        pred_net.external_input.extend(
            ["quantized_data", "weights", "indices", "lengths"]
        )
        pred_net.external_output.append("Y")
        pred_net.op.add().CopyFrom(
            core.CreateOperator(
                "SparseLengthsWeightedSumFused8BitRowwise",
                ["quantized_data", "weights", "indices", "lengths"],
                ["Y"],
            )
        )

        ref_net = caffe2_pb2.NetDef()
        ref_net.name = "ref"
        ref_net.external_input.extend(
            ["quantized_data", "weights", "indices", "lengths"]
        )
        ref_net.external_output.append("Y")
        ref_net.op.add().CopyFrom(
            core.CreateOperator(
                "SparseLengthsWeightedSumFused8BitRowwiseFakeFP32NNPI",
                ["quantized_data", "weights", "indices", "lengths"],
                ["Y"],
            )
        )

        workspace.FeedBlob("data", data)
        workspace.RunOperatorOnce(
            core.CreateOperator(
                "FloatToFused8BitRowwiseQuantized", ["data"], ["quantized_data"]
            )
        )

        quantized_data = workspace.FetchBlob("quantized_data")

        onnxified_net = onnxifi_caffe2_net(
            pred_net,
            {},
            max_batch_size=1,
            max_seq_size=n,
            debug=True,
            adjust_batch=True,
            use_onnx=False,
        )
        workspace.FeedBlob("indices", indices)
        workspace.FeedBlob("lengths", lengths)
        workspace.FeedBlob("weights", weights)

        workspace.CreateNet(onnxified_net)
        workspace.CreateNet(ref_net)

        workspace.RunNet(onnxified_net.name)
        Y_glow = workspace.FetchBlob("Y")

        workspace.RunNet(ref_net.name)
        Y_ref = workspace.FetchBlob("Y")

        diff = np.abs((Y_ref - Y_glow) / (Y_ref + 1e-8))
        max_err = np.max(diff, axis=1)
        num_offenders = (max_err > 0).sum()
        if num_offenders > 0:
            np.set_printoptions(precision=12)
            print(
                "ref",
                Y_ref.astype(np.float16).astype(np.float32),
                "glow",
                Y_glow.astype(np.float16).astype(np.float32),
            )
            print_test_debug_info(
                "test_small_sls_acc32",
                {
                    "seed": seed,
                    "indices": indices,
                    "data": data,
                    "quantized_data": quantized_data,
                    "lengths": lengths,
                    "weights": weights,
                    "Y_glow": Y_glow,
                    "Y_ref": Y_ref,
                    "diff": diff,
                    "rowwise_diff": np.max(diff, axis=1),
                },
            )
            assert 0

    def test_slws_fused_8bit_rowwise_acc32_nnpi(self):
        workspace.GlobalInit(
            [
                "caffe2",
                "--glow_global_fp16=0",
                "--glow_global_fused_scale_offset_fp16=0",
                "--glow_global_force_sls_fp16_accum=0",
            ]
        )
        # Comment out for predictable debugging
        seed = int(time.time() * 1000) % 2 ** 16
        print(seed)
        np.random.seed(seed)
        workspace.ResetWorkspace()

        n = 20000
        DIM = 6
        data = (4 * np.random.random_sample((n, DIM)) + 1).astype(np.float32)

        max_segments = 200
        max_segment_length = 200
        num_lengths = np.random.randint(0, max_segments + 1)
        # number of segments to run
        lengths = np.random.randint(2, max_segment_length + 1, size=num_lengths).astype(
            np.int32
        )
        num_indices = np.sum(lengths)
        indices = np.random.randint(low=0, high=n, size=num_indices, dtype=np.int64)
        weights = np.random.uniform(low=0.01, high=0.5, size=[len(indices)]).astype(
            np.float32
        )

        pred_net = caffe2_pb2.NetDef()
        pred_net.name = "pred"
        pred_net.external_input.extend(
            ["quantized_data", "weights", "indices", "lengths"]
        )
        pred_net.external_output.append("Y")
        pred_net.op.add().CopyFrom(
            core.CreateOperator(
                "SparseLengthsWeightedSumFused8BitRowwise",
                ["quantized_data", "weights", "indices", "lengths"],
                ["Y"],
            )
        )

        ref_net = caffe2_pb2.NetDef()
        ref_net.name = "ref"
        ref_net.external_input.extend(
            ["quantized_data", "weights", "indices", "lengths"]
        )
        ref_net.external_output.append("Y")
        ref_net.op.add().CopyFrom(
            core.CreateOperator(
                "SparseLengthsWeightedSumFused8BitRowwiseFakeFP32NNPI",
                ["quantized_data", "weights", "indices", "lengths"],
                ["Y"],
            )
        )

        workspace.FeedBlob("data", data)
        workspace.RunOperatorOnce(
            core.CreateOperator(
                "FloatToFused8BitRowwiseQuantized", ["data"], ["quantized_data"]
            )
        )
        onnxified_net = onnxifi_caffe2_net(
            pred_net,
            {},
            max_batch_size=max_segments,
            max_seq_size=max_segments * max_segment_length,
            debug=True,
            adjust_batch=True,
            use_onnx=False,
        )
        workspace.FeedBlob("indices", indices)
        workspace.FeedBlob("lengths", lengths)
        workspace.FeedBlob("weights", weights)

        workspace.CreateNet(onnxified_net)
        workspace.CreateNet(ref_net)

        workspace.RunNet(onnxified_net.name)
        Y_glow = workspace.FetchBlob("Y")

        workspace.RunNet(ref_net.name)
        Y_ref = workspace.FetchBlob("Y")

        diff = np.abs((Y_ref - Y_glow) / (Y_ref + 1e-8))
        max_err = np.max(diff, axis=1)
        num_offenders = (max_err > 0).sum()
        if num_offenders > 0:
            print_test_debug_info(
                "test_slws_fused_8bit_rowwise_acc32_nnpi",
                {
                    "indices": indices,
                    "data": data.shape,
                    "lengths": lengths,
                    "weights": weights,
                    "Y_glow": Y_glow,
                    "Y_ref": Y_ref,
                    "diff": diff,
                    "rowwise_diff": np.max(diff, axis=1),
                },
            )
            assert 0
