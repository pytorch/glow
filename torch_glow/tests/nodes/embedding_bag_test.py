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

from __future__ import absolute_import, division, print_function, unicode_literals

import numpy as np
import torch
from tests import utils
from tests.utils import DEFAULT_BACKEND, check_skip


class TestEmbeddingBag(utils.TorchGlowTestCase):
    supported_backends = {"Interpreter", "NNPI"}

    def test_embedding_bag_basic(self):
        """Test of aten::embedding_bag node on glow"""

        check_skip(self)

        class TestModule(torch.nn.Module):
            def forward(self, input, offsets, per_sample_weights):
                weight = torch.FloatTensor([[1, 2.3, 3], [4, 5.1, 6.3]])
                embedding_sum = torch.nn.EmbeddingBag.from_pretrained(
                    weight, mode="sum", include_last_offset=True
                )
                a = embedding_sum(input, offsets)
                b = embedding_sum(input, offsets, per_sample_weights)
                return a, b

        input = torch.LongTensor([1, 0, 0, 1, 1])
        offsets = torch.LongTensor([0, 1, 5])  # final item is endOffset
        per_sample_weights = torch.FloatTensor([1, 2, 3, 4, 5])

        utils.compare_tracing_methods(
            TestModule(),
            input,
            offsets,
            per_sample_weights,
            fusible_ops={"aten::embedding_bag"},
        )


class TestQuantizedEmbeddingBag(utils.TorchGlowTestCase):
    supported_backends = {"Interpreter", "NNPI"}

    @utils.deterministic_expand(
        [
            # explicit local param declaration required for lambda fn with loops for correct param generation
            lambda num_lengths=num_lengths, is4bit=is4bit, is_weighted=is_weighted, use_fp16=use_fp16, per_sample_weights_fp16=per_sample_weights_fp16: (
                "{len}{bits}{weighted}{fp16}{sample_weights}{backend}".format(
                    len=num_lengths,
                    bits="_4bit" if is4bit else "_byte",
                    weighted="_weighted" if is_weighted else "",
                    fp16="_fp16" if use_fp16 else "",
                    sample_weights="_sample_weights_fp16"
                    if per_sample_weights_fp16 and is_weighted
                    else "",
                    backend="_" + DEFAULT_BACKEND,
                ),
                num_lengths,
                is4bit,
                is_weighted,
                use_fp16,
                per_sample_weights_fp16,
            )
            for num_lengths in [0, 8]
            for is4bit in [False, True]
            for is_weighted in [False, True]
            for use_fp16 in [False, True]
            for per_sample_weights_fp16 in [False, True]
        ]
    )
    def test_embedding_bag_rowwise_offsets(
        self,
        name,
        num_lengths,
        is4bit,
        is_weighted,
        use_fp16,
        per_sample_weights_fp16,
    ):
        """Test of quantized::embedding_bag_byte_rowwise_offsets and
        quantized::embedding_bag_4bit_rowwise_offsets node on glow"""
        check_skip(self)

        class TestModule(torch.nn.Module):
            def __init__(self, q_weights, is4bit=False, per_sample_weights=None):
                super().__init__()
                self.q_weights = q_weights
                self.per_sample_weights = per_sample_weights
                if is4bit:
                    self.op = torch.ops.quantized.embedding_bag_4bit_rowwise_offsets
                else:
                    self.op = torch.ops.quantized.embedding_bag_byte_rowwise_offsets

            def forward(self, indices, offsets):
                return self.op(
                    self.q_weights,
                    indices,
                    offsets,
                    mode=0,
                    per_sample_weights=self.per_sample_weights,
                    include_last_offset=True,
                )

        # generate random weights and indices
        num_embeddings = 16
        embedding_dim = 4
        weights = torch.from_numpy(
            (np.random.random_sample((num_embeddings, embedding_dim)) + 1).astype(
                np.float32
            )
        )
        q_weights = (
            torch.ops.quantized.embedding_bag_4bit_prepack(weights)
            if is4bit
            else torch.ops.quantized.embedding_bag_byte_prepack(weights)
        )
        np_lengths = (
            np.zeros(shape=[10]).astype(np.int32)
            if num_lengths == 0
            else np.random.randint(0, num_lengths, size=10).astype(np.int32)
        )
        num_lengths = np.sum(np_lengths)
        lengths = torch.from_numpy(np_lengths)
        indices = torch.from_numpy(
            np.random.randint(
                low=0, high=num_embeddings, size=num_lengths, dtype=np.int64
            )
        ).long()
        offsets = torch.cat([torch.zeros([1]), torch.cumsum(lengths, 0)]).long()

        per_sample_weights_type = (
            np.float16 if per_sample_weights_fp16 and is4bit else np.float32
        )
        per_sample_weights = torch.from_numpy(
            np.random.uniform(low=0.01, high=0.5, size=[len(indices)]).astype(
                per_sample_weights_type
            )
        )

        m = TestModule(q_weights, is4bit, per_sample_weights if is_weighted else None)

        utils.compare_tracing_methods(
            m,
            indices,
            offsets,
            fusible_ops={
                "quantized::embedding_bag_4bit_rowwise_offsets"
                if is4bit
                else "quantized::embedding_bag_byte_rowwise_offsets"
            },
            fp16=use_fp16,
            # FP16 version is known to yeild different results, so our
            # test here is mainly focusing on the flow rather than actual
            # accuracy. There will be additional coverage on accuracy of
            # the lowered modules
            atol=0.02 if (is4bit or use_fp16) else 5e-4,
        )
