#!/usr/bin/env python2
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

import shutil
import sys
import tempfile

import argparse
import os
import pexpect

import PIL.Image as Image
import torchvision

parser = argparse.ArgumentParser(
    description="Glow image-classifier Driver for "
    "TopK ImageNet Calculation")

parser.add_argument(
    "--validation-images-dir",
    metavar="DIR",
    required=True,
    help="Path to the directory containing the validation set "
    "of images. Subdirectories are expected to be organized "
    "such that when sorted their index corresponds to their "
    "label. For example, if the validation_images_dir contains "
    "{'abc/', 'def/', 'ghi/'}, then this should correspond to "
    "labels {0, 1, 2} respectively.")

parser.add_argument("--batch-size", default=1, type=int, metavar="N",
                    help="Batch size for use with the model. The total number "
                    "of images in the validation_images_dir should be "
                    "divisible by the batch size.")

parser.add_argument("--only-resize-and-save", default=False,
                    action="store_true", help="Use to pre-process images "
                    "to 224x224. Saves the images to "
                    "the validation_images_dir/processed/")

parser.add_argument("--resize-input-images", default=False,
                    action="store_true", help="Resize and center-crop images "
                    "at runtime to 224x224.")

parser.add_argument("--verbose", default=False,
                    action="store_true", help="Verbose printing.")

parser.add_argument("--image-classifier-cmd", default="",
                    help="Command to use for running the image-classifier, "
                    "including the binary and all of its command lime "
                    "parameters.")

# Opens and returns an image located at @param path using the PIL loader.


def pil_loader(path):
    # open path as file to avoid ResourceWarning
    # (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, "rb") as img:
        img = Image.open(img)
        return img.convert("RGB")

# Opens and returns an image located at @param path using the accimage loader.


def accimage_loader(path):
    import accimage
    try:
        return accimage.Image(path)
    except IOError:
        # Potentially a decoding problem, fall back to PIL.Image
        return pil_loader(path)


# Opens and returns an image located at @param path using either the accimage
# loader or PIL loader.
def default_image_loader(path):
    if torchvision.get_image_backend() == "accimage":
        return accimage_loader(path)
    return pil_loader(path)


def get_sorted_img_subdirs(validation_images_dir):
    img_dir_paths = []
    for img_dir in os.listdir(validation_images_dir):
        dir_path = os.path.join(validation_images_dir, img_dir)
        if os.path.isdir(dir_path):
            img_dir_paths.append(img_dir)
    img_dir_paths.sort()

    return img_dir_paths


# @returns two lists of the same length found in directory
# @param validation_images_dir; the first list contains paths to all images
# found, and the second list contains the corresponding labels of the image.
def get_img_paths_and_labels(validation_images_dir):
    img_subdirs = get_sorted_img_subdirs(validation_images_dir)

    # Create lists holding paths to each image to be classified and the label
    # for that image.
    img_paths = []
    img_labels = []
    curr_label_idx = 0
    for img_subdir in img_subdirs:
        img_subdir_path = os.path.join(validation_images_dir, img_subdir)
        for img in os.listdir(img_subdir_path):
            full_img_path = os.path.join(img_subdir_path, img)
            if os.path.isfile(full_img_path):
                img_paths.append(full_img_path)
                img_labels.append(curr_label_idx)
        curr_label_idx = curr_label_idx + 1
    return img_paths, img_labels

# Given an image located at @param img_path, transform the image
# and save it to the path @param path_to_new_img.


def resize_and_save_image(img_path, path_to_new_img):
    # Load the image.
    img = default_image_loader(img_path)

    # Use to Resize and CenterCrop the images to 224x224.
    transform_resize = torchvision.transforms.Compose([
        torchvision.transforms.Resize(256),
        torchvision.transforms.CenterCrop(224),
    ])

    resized_img = transform_resize(img)
    resized_img.save(path_to_new_img, format="png")

# Used to pre-process an input set of images. Takes a string of a directory
# @param validation_images_dir and saves the cropped subset of the images in a
# subdirectory `processed/`, which must not yet exist.


def save_centered_cropped_dataset(validation_images_dir):
    processed_validation_images_dir = os.path.join(validation_images_dir,
                                                   "processed")
    print "Saving centered cropped input images: %s" % (
          processed_validation_images_dir)

    img_subdirs = get_sorted_img_subdirs(validation_images_dir)

    try:
        os.makedirs(processed_validation_images_dir)
    except OSError:
        sys.exit("New validation directory must not exist")

    # Iterate over all labels subdirectories, loading, transforming and saving
    # all images to the new location.
    for img_subdir in img_subdirs:
        orig_img_subdir_path = os.path.join(validation_images_dir, img_subdir)
        processed_img_subdir_path = os.path.join(
            processed_validation_images_dir, img_subdir)

        # Create a new subdirectory for the next label.
        try:
            os.makedirs(processed_img_subdir_path)
        except OSError:
            sys.exit("New label subdirectory somehow already existed.")

        # Transform and save all images in this label subdirectory.
        for orig_img_filename in os.listdir(orig_img_subdir_path):
            orig_img_path = os.path.join(
                orig_img_subdir_path, orig_img_filename)
            if os.path.isfile(orig_img_path):
                processed_img_path = os.path.join(processed_img_subdir_path,
                                                  orig_img_filename)
                resize_and_save_image(orig_img_path, processed_img_path)

# @returns a list of strings (of length equal to the @param batch_size) which
# are paths to images to do inference on. @param img_paths is the set of all
# image paths, @param img_index is the next index to use in @param img_paths,
# and @param tmp_dir_name is the location of where to save the images if
# @param resize_input_images is true. Note that if @param resize_input_images is
# true, then names for the temporary images are used for every batch, thus only
# @param batch_size temporary images will ever exist in @param tmp_dir_name.


def get_curr_img_paths(img_paths, img_index, batch_size, tmp_dir_name,
                       resize_input_images):
    curr_img_paths = []
    for batch_idx in xrange(batch_size):
        img_path = img_paths[img_index + batch_idx]
        # If we are resizing the image then we are going to save it to a
        # temp location to read in later for inference.
        if resize_input_images:
            # Save the new image to the tmp directory. Note that these names are
            # reused every call to get_curr_img_paths().
            path_to_tmp_img = os.path.join(tmp_dir_name, "tmp" +
                                           str(batch_idx) + ".png")
            resize_and_save_image(img_path, path_to_tmp_img)
            img_path = path_to_tmp_img

        curr_img_paths.append(img_path)

    return curr_img_paths

# Verifies that the @param image_classifier_cmd is well formatted via
# assertions.


def verify_spawn_cmd(image_classifier_cmd):
    split_cmd = image_classifier_cmd.split()
    if "image-classifier" in split_cmd[0]:
        assert "-" in split_cmd, "Streaming mode must be used."
        assert "-topk=5" in split_cmd, "-topk=5 must be used."
        assert any("-model-input-name=" in s for s in split_cmd), (
            "image-classifier requires -model-input-name to be specified.")
        assert any("-m=" in s for s in split_cmd), (
            "image-classifier requires -m to be specified")
        assert any("-image-mode=" in s for s in split_cmd), (
            "image-classifier requires -image-mode to be specified")

# Prints the Top-1 and Top-5 accuracy given @param total_image_count, @param
# top1_count, and @param top5_count.


def print_topk_accuracy(total_image_count, top1_count, top5_count):
    top1_accuracy = float(top1_count) / float(total_image_count)
    top5_accuracy = float(top5_count) / float(total_image_count)
    print "\tTop-1 accuracy: " + "{0:.4f}".format(top1_accuracy)
    print "\tTop-5 accuracy: " + "{0:.4f}".format(top5_accuracy)

# Calculates and prints top-1 and top-5 accuracy for images located in
# subdirectories at @param validation_images_dir, given the command line
# parameters passed in to @param args.


def calculate_top_k(validation_images_dir, image_classifier_cmd, batch_size,
                    resize_input_images, verbose):
    print "Calculating Top-1 and Top-5 accuracy..."

    verify_spawn_cmd(image_classifier_cmd)

    img_paths, img_labels = get_img_paths_and_labels(validation_images_dir)

    total_image_count = len(img_paths)

    assert total_image_count % batch_size == 0, (
        "Total number of images must be divisible by batch size")

    if verbose:
        print "Running image classifier with: " + image_classifier_cmd

    try:
        # Create a temporary directory to store the transformed image we
        # classify (if applicable) and the log of image-classifer output.
        tmp_dir_name = tempfile.mkdtemp()
        path_to_tmp_log = os.path.join(tmp_dir_name, "log.txt")
        fout = file(path_to_tmp_log, "w")

        classifier_proc = pexpect.spawn(image_classifier_cmd, logfile=fout,
                                        timeout=None)

        if verbose:
            print "Temp log located at: " + path_to_tmp_log

        prompt = "Enter image filenames to classify: "
        top1_count = 0
        top5_count = 0

        # Process the images in batches as specified on the command line.
        for img_index in xrange(0, total_image_count, batch_size):
            curr_img_paths = get_curr_img_paths(img_paths, img_index,
                                                batch_size, tmp_dir_name,
                                                resize_input_images)

            # Expect prompt from the image-classifier for the next image path.
            classifier_proc.expect(prompt)

            appended_paths = " ".join(curr_img_paths)
            assert len(appended_paths) <= 1024, (
                "Line length is too long (max 1024): %r" % len(appended_paths))

            # Send the paths to the image-classifier.
            classifier_proc.sendline(appended_paths)

            for batch_idx in xrange(batch_size):
                # Now we expect the image-classifier's response with the label.
                # The first line will include the path to the file, e.g.:
                #  File: tests/images/imagenet/cat_285.png
                classifier_proc.expect(" File: " + curr_img_paths[batch_idx])

                # All labels will be formatted like:
                # Label-K1: 281 (probability: 0.7190)
                top5_labels = []
                for _ in xrange(5):
                    label_and_prob = classifier_proc.readline()
                    # Get the label from the line.
                    label = label_and_prob.split()[1]
                    top5_labels.append(int(label))

                expected_label = img_labels[img_index + batch_idx]
                if expected_label == top5_labels[0]:
                    top1_count += 1
                if expected_label in top5_labels:
                    top5_count += 1

            curr_completed_count = img_index + batch_size
            if curr_completed_count % 100 == 0:
                print "Finished image index %d out of %d" % (
                    (curr_completed_count, total_image_count))
                if verbose:
                    print "  Current Top-1/5 accuracy:"
                    print_topk_accuracy(curr_completed_count, top1_count,
                                        top5_count)
                else:
                    print ""

    finally:
        classifier_proc.close(force=True)

        # Remove the temp directory we used to save the images and log.
        shutil.rmtree(tmp_dir_name)

    print "\nCompleted running; Final Top-1/5 accuracy across %d images:" % (
        total_image_count)
    print_topk_accuracy(total_image_count, top1_count, top5_count)


def main():
    # Parse the recognized command line arguments into args.
    args = parser.parse_args()

    # Path to the directory containing the validation set of images.
    # Subdirectories are expected to be organized such that when sorted their
    # index corresponds to their label. For example, if the
    # validation_images_dir contains {'abc/', 'def/', 'ghi/'}, then this should
    # correspond to labels {0, 1, 2} respectively.
    validation_images_dir = os.path.join(args.validation_images_dir)
    assert os.path.exists(validation_images_dir), (
        "Validation directory does not exist: " + validation_images_dir)

    # This is used solely to pre-process the input image set.
    if args.only_resize_and_save:
        save_centered_cropped_dataset(validation_images_dir)
        return

    calculate_top_k(validation_images_dir, args.image_classifier_cmd,
                    args.batch_size, args.resize_input_images, args.verbose)


if __name__ == "__main__":
    main()
