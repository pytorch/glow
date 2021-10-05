# isort:skip_file
from __future__ import absolute_import, division, print_function, unicode_literals

import torch
import torch.jit
import torch_glow
from tests import utils
from tests import utils


# Use a model containing quantized::conv2d to verify preprocessed module is
# save correctly in a lowered module (ops with packed weights like this one
# are rewirtten during lowering, therefore should only be present in the
# original graph).
class Bar(torch.nn.Module):
    def __init__(self):
        super(Bar, self).__init__()
        with torch.no_grad():
            conv = torch.nn.Conv2d(4, 2, [2, 2], groups=1)
            conv.weight.random_(-1, 1)
            conv.bias.data.random_(-1, 1)
            self.model = torch.ao.quantization.QuantWrapper(conv)
            self.model.qconfig = torch.ao.quantization.get_default_qconfig("fbgemm")
            torch.ao.quantization.prepare(self.model, inplace=True)
            torch.ao.quantization.convert(self.model, inplace=True)

    def forward(self, x):
        return self.model(x)


class TestToGlowSavePreprocessedModule(utils.TorchGlowTestCase):
    def test_save_preprocessed_module(self):
        with torch.no_grad():
            x = torch.randn([1, 4, 4, 4], dtype=torch.float32)
            model = Bar()
            model.eval()
            model = torch.jit.trace(model, x)

            spec = torch_glow.CompilationSpec()
            spec.get_settings().set_glow_backend("Interpreter")

            compilation_group = torch_glow.CompilationGroup()
            spec.compilation_groups_append(compilation_group)

            compilation_group.input_sets_append(
                torch_glow.input_specs_from_tensors([x])
            )

            torch_glow.disableFusionPass()
            torch_glow.enable_convert_to_fp16()
            glow_mod = torch_glow.to_glow(model, spec)

            reloaded = utils.save_and_reload_model(glow_mod)

            attrname = "__processed_module"
            pp = getattr(reloaded._c, attrname)
            pt_model = torch.jit._recursive.wrap_cpp_module(pp)
            graph = pt_model.graph_for(x)
            found = False
            for node in graph.nodes():
                if node.kind() == "quantized::conv2d":
                    found = True

            assert found
