# isort:skip_file
from __future__ import absolute_import, division, print_function, unicode_literals

import torch
import torch_glow
from tests import utils
from tests.utils import GLOW_FUSION_GROUP


class TestQuantizedCut(utils.TorchGlowTestCase):
    def test_quantized_cut(self):
        """Test cut quantized chunk in the middle."""
        torch._C._jit_set_profiling_executor(False)
        torch._C._jit_set_profiling_mode(False)

        def fun(a, b, c, d):
            q = torch.nn.quantized.Quantize(
                scale=1.0 / 21, zero_point=0, dtype=torch.quint8
            )
            dq = torch.nn.quantized.DeQuantize()
            a = q(a)
            b = q(b)
            c = q(c)
            d = q(d)
            adds = torch.ops.quantized.add(a, b, scale=1.0 / 17, zero_point=5)
            adds2 = torch.ops.quantized.add(c, d, scale=1.0 / 14, zero_point=4)
            res = torch.ops.quantized.add_relu(
                adds, adds2, scale=1.0 / 18, zero_point=6
            )
            res = torch.ops.quantized.add(res, res, scale=1.0 / 13, zero_point=7)
            res = dq(res)
            return res

        with torch.no_grad():
            a = torch.randn([5, 5])
            b = torch.randn([5, 5])
            c = torch.randn([5, 5])
            d = torch.randn([5, 5])
            res_torch = fun(a, b, c, d)
            torch_glow.enableFusionPass_DO_NOT_USE_THIS()
            # Cut using blocklist functionality
            blocklist = ["quantized::add_relu"]
            torch_glow.setFusionBlocklist(blocklist)
            torch_glow.setGlowBackend("Interpreter")
            traced_model = torch.jit.trace(fun, (a, b, c, d))
            for node in traced_model.graph_for(a, b, c, d).nodes():
                kind = node.kind()
                # Make sure the blocklist is working
                assert (
                    kind == GLOW_FUSION_GROUP
                    or kind in blocklist
                    or kind == "prim::Constant"
                )
            res_glow = traced_model(a, b, c, d)
            print(res_torch)
            print(res_glow)
            assert torch.allclose(res_torch, res_glow)
