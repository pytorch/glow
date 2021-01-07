from __future__ import absolute_import, division, print_function, unicode_literals

import unittest

import torch
import torchvision
from parameterized import parameterized
from tests import utils


def gen_boxes_scores(size):
    boxes = torch.rand(size, 4)
    boxes[:, 2:] = torch.ones(size, 2)
    scores = torch.rand(size)
    return (boxes, scores)


class SimpleNmsModule(torch.nn.Module):
    def __init__(self, max_outputs):
        super(SimpleNmsModule, self).__init__()
        self.max_outputs = max_outputs

    def forward(self, boxes, scores, iou_threshold):
        keep = torchvision.ops.nms(boxes, scores, iou_threshold)
        # For the results to match, we have to slice the output
        # as the output of glow is padded
        return keep[: self.max_outputs]


class TestNms(unittest.TestCase):
    def test_nms_basic(self):
        boxes, scores = gen_boxes_scores(5)
        iou_threshold = torch.tensor(0.3)
        # Get the size of the output of torch nms
        max_outputs = torchvision.ops.nms(boxes, scores, iou_threshold).shape[0]

        utils.compare_tracing_methods(
            SimpleNmsModule(max_outputs),
            boxes,
            scores,
            iou_threshold,
            fusible_ops={"torchvision::nms"},
        )

    def test_nms_large_input(self):
        boxes, scores = gen_boxes_scores(500)
        iou_threshold = torch.tensor(0.5)
        # Get the size of the output of torch nms
        max_outputs = torchvision.ops.nms(boxes, scores, iou_threshold).shape[0]

        utils.compare_tracing_methods(
            SimpleNmsModule(max_outputs),
            boxes,
            scores,
            iou_threshold,
            fusible_ops={"torchvision::nms"},
        )
