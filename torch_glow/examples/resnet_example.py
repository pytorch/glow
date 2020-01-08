from __future__ import absolute_import, division, print_function, unicode_literals

import torch
import torch_glow
from PIL import Image

import utils.torchvision_fake.transforms as torchvisionTransforms
import utils.torchvision_fake.resnet as resnet

import argparse


def load_image(image_path):
    image = Image.open(image_path).convert('RGB')
    transformed_image = transform_image(image)
    return torch.reshape(transformed_image, (1, 3, 224, 224))


def transform_image(image):
    """
    Given a PIL image, transform it to a normalized tensor for classification.
    """
    image = torchvisionTransforms.resize(image, 256)
    image = torchvisionTransforms.center_crop(image, 224)
    image = torchvisionTransforms.to_tensor(image)
    image = torchvisionTransforms.normalize(
        image, mean=[
            0.485, 0.456, 0.406], std=[
            0.229, 0.224, 0.225])
    return image


def run_model(model, image, use_glow, backend, print_graph):
    if use_glow:
        torch_glow.enableFusionPass()
        if backend:
            torch_glow.setGlowBackend(backend)

    with torch.no_grad():
        traced = torch.jit.trace(model, image)
        if print_graph:
            print(traced.graph_for(image))
        all_outputs = traced(image)
        topk = all_outputs.topk(5)
        return(topk[1], topk[0])


def run():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", type=str, required=True,
                        help="Location of the image to be classified")
    parser.add_argument(
        "--k",
        type=int,
        default=5,
        help="how many results to show")
    parser.add_argument(
        "--skip_glow",
        action='store_true',
        default=False,
        help="Don't run using Glow")
    parser.add_argument(
        "--print_graph",
        action='store_true',
        default=False,
        help="Don't run using Glow")
    parser.add_argument(
        "--backend",
        action="store",
        default=None,
        help="Select Glow backend to run. Default is not to request a specific backend.")
    args = parser.parse_args()

    image = load_image(args.image)
    model = resnet.resnet18(pretrained=True, progress=True)
    model.eval()
    use_glow = not args.skip_glow

    (indices, scores) = run_model(model, image,
                                  use_glow=use_glow,
                                  backend=args.backend,
                                  print_graph=args.print_graph)
    print("rank", "class", "P")
    for i in range(args.k):
        print(i, int(indices[0][i]), float(scores[0][i]))


run()
