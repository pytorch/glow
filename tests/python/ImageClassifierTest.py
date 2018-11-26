import os
import subprocess
import sys
import unittest


class Image:
    def __init__(self, filename, labels):
        self.filename = filename
        self.labels = labels


class Model:
    def __init__(
        self,
        name,
        dataset,
        input_name,
        image_mode,
        imagenet_norm=False,
        batching=True,
        compute_softmax=False,
        image_layout=None,
        label_offset=None,
    ):
        self.name = name
        self.dataset = dataset
        self.input_name = input_name
        self.image_mode = image_mode
        self.imagenet_norm = imagenet_norm
        self.batching = batching
        self.compute_softmax = compute_softmax
        self.image_layout = image_layout
        self.label_offset = label_offset

    def command_line(self, images):
        return (
            ["./bin/image-classifier"]
            + images
            + (["-use-imagenet-normalization"] if self.imagenet_norm else [])
            + ["-image_mode", self.image_mode]
            + ["-m", self.name]
            + ["-model_input_name", self.input_name]
            + (["-compute_softmax"] if self.compute_softmax else [])
            + (["-image_layout", self.image_layout] if self.image_layout else [])
            + (["-label_offset", self.label_offset] if self.label_offset else [])
        )

    def check_output(self, images):
        return subprocess.check_output(self.command_line(images)).decode("ascii")

    def image_path(self, image):
        return os.path.join("./tests/images", self.dataset, image.filename)

    def images(self):
        return [self.image_path(img) for img in images[self.dataset]]

    def execute(self, testcase):
        if self.batching:
            out = self.check_output(self.images())
        else:
            out = "".join(self.check_output([image]) for image in self.images())
        predictions = [
            line.split()[3] for line in out.split("\n") if line.startswith(" File:")
        ]
        for prediction, image in zip(predictions, images[self.dataset]):
            testcase.assertIn(int(prediction), image.labels)


images = {
    "imagenet": [
        Image("cat_285.png", [285, 281]),
        Image("dog_207.png", [207]),
        Image("zebra_340.png", [340]),
    ],
    "imagenet_299": [
        Image("cat_285_299.png", [285, 281]),
        Image("dog_207_299.png", [207]),
        Image("zebra_340_299.png", [340]),
    ],
    "mnist": [
        Image("0_1009.png", [0]),
        Image("1_1008.png", [1]),
        Image("2_1065.png", [2]),
        Image("3_1020.png", [3]),
        Image("4_1059.png", [4]),
        Image("5_1087.png", [5]),
        Image("6_1099.png", [6]),
        Image("7_1055.png", [7]),
        Image("8_1026.png", [8]),
        Image("9_1088.png", [9]),
    ],
}


models = [
    Model(
        name="resnet50",
        dataset="imagenet",
        input_name="gpu_0/data",
        image_mode="0to1",
        imagenet_norm=True,
    ),
    Model(
        name="vgg19", dataset="imagenet", input_name="data", image_mode="neg128to127"
    ),
    Model(
        name="squeezenet",
        dataset="imagenet",
        input_name="data",
        image_mode="neg128to127",
    ),
    Model(
        name="zfnet512",
        dataset="imagenet",
        input_name="gpu_0/data",
        image_mode="0to255",
    ),
    Model(
        name="densenet121",
        dataset="imagenet",
        input_name="data",
        image_mode="0to1",
        compute_softmax=True,
    ),
    Model(
        name="shufflenet",
        dataset="imagenet",
        input_name="gpu_0/data",
        image_mode="0to1",
    ),
    Model(name="lenet_mnist", dataset="mnist", input_name="data", image_mode="0to1"),
    Model(
        name="inception_v1", dataset="imagenet", input_name="data", image_mode="0to255"
    ),
    Model(
        name="bvlc_alexnet", dataset="imagenet", input_name="data", image_mode="0to255"
    ),
    Model(
        name="resnet50/model.onnx",
        dataset="imagenet",
        input_name="gpu_0/data_0",
        image_mode="0to1",
        imagenet_norm=True,
        batching=False,
    ),
    Model(
        name="vgg19/model.onnx",
        dataset="imagenet",
        input_name="data",
        image_mode="neg128to127",
        batching=False,
    ),
    Model(
        name="squeezenet/model.onnx",
        dataset="imagenet",
        input_name="data",
        image_mode="neg128to127",
        batching=False,
    ),
    Model(
        name="zfnet512/model.onnx",
        dataset="imagenet",
        input_name="gpu_0/data",
        image_mode="0to255",
        batching=False,
    ),
    Model(
        name="densenet121/model.onnx",
        dataset="imagenet",
        input_name="data",
        image_mode="0to1",
        compute_softmax=True,
        batching=False,
    ),
    Model(
        name="shufflenet/model.onnx",
        dataset="imagenet",
        input_name="gpu_0/data",
        image_mode="0to1",
        batching=False,
    ),
    Model(
        name="mnist.onnx",
        dataset="mnist",
        input_name="data",
        image_mode="0to1",
        batching=False,
    ),
    Model(
        name="inception_v1/model.onnx",
        dataset="imagenet",
        input_name="data",
        image_mode="0to255",
        batching=False,
    ),
    Model(
        name="bvlc_alexnet/model.onnx",
        dataset="imagenet",
        input_name="data",
        image_mode="0to255",
        batching=False,
    ),
    Model(
        name="googlenet_v1_slim/googlenet_v1_slim.onnx",
        dataset="imagenet_299",
        input_name="input:0",
        image_mode="0to1",
        image_layout="NHWC",
        label_offset="1",
    ),
    Model(
        name="googlenet_v4_slim/googlenet_v4_slim.onnx",
        dataset="imagenet_299",
        input_name="input:0",
        image_mode="0to1",
        image_layout="NHWC",
        label_offset="1",
    ),
]


class TestModels(unittest.TestCase):
    def test_models(self):
        for model in models:
            model.execute(self)


if __name__ == "__main__":
    unittest.main()
