# isort:skip_file

# pyre-ignore-all-errors
from __future__ import absolute_import, division, print_function, unicode_literals

import pickle

import torch
import torch_glow
from glow.glow.torch_glow.tests.tests import utils


class TestCompilationSpec(utils.TorchGlowTestCase):
    def build_compiliation_spec(self):
        compilation_spec = torch_glow.CompilationSpec()

        compilation_spec_settings = compilation_spec.get_settings()
        compilation_spec_settings.set_glow_backend("CPU")
        compilation_spec_settings.set_enable_fuser(True)

        fuser_settings = compilation_spec.get_fuser_settings()
        fuser_settings.set_min_fusion_group_size(3)
        fuser_settings.set_max_fusion_merge_size(4)
        fuser_settings.set_fusion_start_index(5)
        fuser_settings.set_fusion_end_index(6)
        fuser_settings.op_blacklist_append("aten::mean")
        fuser_settings.op_blacklist_append("aten::dropout")

        compilation_group = torch_glow.CompilationGroup()

        input1_spec = torch_glow.input_spec_from_tensor(torch.randn(2, 3, 224, 224))
        input2_spec = torch_glow.input_spec_from_tensor(
            torch.randn(3, 2).to(torch.float16)
        )
        compilation_group.input_sets_append([input1_spec, input2_spec])
        compilation_group.input_sets_append(
            torch_glow.input_specs_from_tensors(
                [torch.randn(1, 3, 224, 224), torch.randn(4, 1)]
            )
        )

        compilation_group_settings = compilation_group.get_settings()
        compilation_group_settings.set_convert_to_fp16(True)
        compilation_group_settings.set_num_devices_to_use(50)
        compilation_group_settings.set_replication_count(52)
        compilation_group_settings.backend_specific_opts_insert("apple", "orange")

        compilation_spec.compilation_groups_append(compilation_group)

        default_compilation_group_settings = (
            compilation_spec.get_default_compilation_group_settings()
        )
        default_compilation_group_settings.set_convert_to_fp16(False)
        default_compilation_group_settings.set_num_devices_to_use(89)
        default_compilation_group_settings.set_replication_count(90)
        default_compilation_group_settings.backend_specific_opts_insert(
            "hello", "goodbye"
        )

        return compilation_spec

    def validate_compilation_spec(self, compilation_spec):
        compilation_spec_settings = compilation_spec.get_settings()
        self.assertEqual(compilation_spec_settings.get_glow_backend(), "CPU")
        self.assertEqual(compilation_spec_settings.get_enable_fuser(), True)

        fuser_settings = compilation_spec.get_fuser_settings()
        self.assertEqual(fuser_settings.get_min_fusion_group_size(), 3)
        self.assertEqual(fuser_settings.get_max_fusion_merge_size(), 4)
        self.assertEqual(fuser_settings.get_fusion_start_index(), 5)
        self.assertEqual(fuser_settings.get_fusion_end_index(), 6)
        self.assertEqual(fuser_settings.get_op_blacklist()[0], "aten::mean")
        self.assertEqual(fuser_settings.get_op_blacklist()[1], "aten::dropout")

        compilation_groups = compilation_spec.get_compilation_groups()
        self.assertEqual(len(compilation_groups), 1)

        compilation_group = compilation_groups[0]

        input_sets = compilation_group.get_input_sets()
        self.assertEqual(len(input_sets), 2)

        self.assertEqual(input_sets[0][0].get_dims(), [2, 3, 224, 224])
        self.assertEqual(input_sets[0][1].get_dims(), [3, 2])
        self.assertEqual(input_sets[1][0].get_dims(), [1, 3, 224, 224])
        self.assertEqual(input_sets[1][1].get_dims(), [4, 1])

        # 5 is at::Half
        self.assertEqual(input_sets[0][1].get_elem_type(), 5)

        compilation_group_settings = compilation_group.get_settings()
        self.assertEqual(compilation_group_settings.get_convert_to_fp16(), True)
        self.assertEqual(compilation_group_settings.get_num_devices_to_use(), 50)
        self.assertEqual(compilation_group_settings.get_replication_count(), 52)
        self.assertEqual(
            compilation_group_settings.backend_specific_opts_at("apple"), "orange"
        )

        default_compilation_group_settings = (
            compilation_spec.get_default_compilation_group_settings()
        )

        self.assertEqual(
            default_compilation_group_settings.get_convert_to_fp16(), False
        )
        self.assertEqual(
            default_compilation_group_settings.get_num_devices_to_use(), 89
        )
        self.assertEqual(default_compilation_group_settings.get_replication_count(), 90)
        self.assertEqual(
            default_compilation_group_settings.backend_specific_opts_at("hello"),
            "goodbye",
        )

    def test_new_glow_compile_spec(self):
        """Test glow compile spec basics."""

        compilation_spec = self.build_compiliation_spec()

        # Sanity check
        self.validate_compilation_spec(compilation_spec)

        # Serialize and deserialize
        pickled = pickle.dumps(compilation_spec)
        unpickled = pickle.loads(pickled)

        # Recheck the spec
        self.validate_compilation_spec(unpickled)
