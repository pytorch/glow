from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import numpy as np
import glow.fb.test.init_shared_libs  # noqa

from caffe2.proto import caffe2_pb2
from caffe2.python import core, workspace
from caffe2.python.onnx.onnxifi import onnxifi_caffe2_net
from glow.fb.test.test_utils import print_test_debug_info
from hypothesis import given
import hypothesis.strategies as st
import caffe2.python.hypothesis_test_util as hu
import caffe2.python.serialized_test.serialized_test_util as serial

core.GlobalInit(["caffe2", "--caffe2_log_level=-3", "--glow_global_fp16=1"])

GLOW_MATMUL_RTOL = 1e-3


class TestBatchMatMul(serial.SerializedTestCase):
    # @settings(max_examples=30)
    @given(
        #C=0, #st.integers(min_value=0, max_value=3),  # number of batch dims
        M=st.integers(min_value=1, max_value=10),
        K=st.integers(min_value=1, max_value=10),
        N=st.integers(min_value=1, max_value=10),
        trans_a=st.booleans(),
        trans_b=st.booleans(),
        run_ints=st.booleans(),
        **hu.gcs
    )
    def test_batch_matmul(self, M, K, N, trans_a, trans_b, run_ints, gc, dc):
        workspace.ResetWorkspace()
        C = 0  # TODO
        batch_dims = np.random.randint(
            low=1,
            high=3,
            size=C,
            dtype=np.int64).tolist()

        if run_ints:
            X = np.random.randint(low=1, high=3, size=((1, M, K))).astype(np.float32)
        else:
            X = 100 * (np.random.rand(*(batch_dims + [M, K])).astype(np.float32) - 0.5)
        if trans_a:
            X = X.swapaxes(-1, -2)

        if run_ints:
            Y = np.random.randint(low=1, high=3, size=((1, K, N))).astype(np.float32)
        else:
            Y = 100 * (np.random.rand(*(batch_dims + [K, N])).astype(np.float32) - 0.5)
        if trans_b:
            Y = Y.swapaxes(-1, -2)

        pred_net = caffe2_pb2.NetDef()
        pred_net.name = "pred"
        pred_net.external_input.extend(["X", "Y"])
        pred_net.external_output.append("out")
        pred_net.op.add().CopyFrom(
            core.CreateOperator(
                'BatchMatMul', ['X', 'Y'], 'out', trans_a=trans_a, trans_b=trans_b
            )
        )

        pred_net_ref = core.Net("pred_net_ref")
        pred_net_ref.BatchMatMulFP16Acc16Fake(
            ["X", "Y"], ['out'], trans_a=trans_a, trans_b=trans_b)

        print("dims", batch_dims, X.shape, Y.shape)
        pred_net_onnxified = onnxifi_caffe2_net(pred_net,
                                                {"X": X.shape, "Y": Y.shape},
                                                debug=True,
                                                adjust_batch=False,
                                                use_onnx=False)
        num_onnxified_ops = sum(
            1 if o.type == "Onnxifi" else 0 for o in pred_net_onnxified.op)
        np.testing.assert_equal(num_onnxified_ops, 1)

        workspace.FeedBlob("X", X)
        workspace.FeedBlob("Y", Y)
        workspace.CreateNet(pred_net_onnxified)
        workspace.CreateNet(pred_net_ref)

        # Run Glow net
        workspace.RunNet(pred_net_onnxified.name)
        out_glow = workspace.FetchBlob('out')

        # Run caffe2 net
        workspace.RunNet(pred_net_ref)
        out_c2_fakefp16 = workspace.FetchBlob('out')

        diff = np.abs((out_c2_fakefp16 - out_glow) / (out_c2_fakefp16 + 1e-8))
        rowdiff = np.max(diff, axis=1)

        success = True
        if run_ints:
            if not np.allclose(out_glow, out_c2_fakefp16):
                success = False
        else:
            n_offenders = np.count_nonzero(rowdiff[rowdiff > GLOW_MATMUL_RTOL])
            # Find the max difference per row, if more than 10% of the rows
            # are bigger, consider it a failure.
            if n_offenders * 10 > rowdiff.shape[0]:
                success = False

        if not success:
            print_test_debug_info("bmm",
                {"m": M, "k": K, "n": N, "X": X, "Y": Y,
                 "out_glow": out_glow,
                 "out_c2_fakefp16": out_c2_fakefp16,
                 "diff": diff})
            assert(0)


if __name__ == "__main__":
    import unittest
    unittest.main()
