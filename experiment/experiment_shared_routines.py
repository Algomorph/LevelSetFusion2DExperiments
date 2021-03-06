#  ================================================================
#  Created by Gregory Kramida on 11/29/18.
#  Copyright (c) 2018 Gregory Kramida
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#  http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
#  ================================================================

# contains some routines shared by the single-frame and multi-frame experiments

# stdlib
from enum import Enum
import re
import os

# libraries
import cv2
import numpy as np

# local
import experiment.dataset as dts
import utils.path as pu


def is_unmasked_image_row_empty(path, ix_row):
    image = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    return np.sum(image[ix_row]) == 0


def is_masked_image_row_empty(image_path, mask_path, ix_row):
    image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
    mask = cv2.imread(mask_path, cv2.IMREAD_UNCHANGED)
    image[mask == 0] = 0
    return np.sum(image[ix_row]) == 0


def is_image_row_empty(image_path, mask_path, ix_row, check_masked):
    if check_masked:
        return is_masked_image_row_empty(image_path, mask_path, ix_row)
    else:
        return is_unmasked_image_row_empty(image_path, ix_row)


class FrameFilenameFormat(Enum):
    FIVE_DIGIT = 0
    SIX_DIGIT = 1


def check_frame_count_and_format(frames_path, turn_mask_checking_off=False):
    depth_five_digit_pattern = re.compile(r"^depth\_\d{5}[.]png$")
    depth_six_digit_pattern = re.compile(r"^depth\_\d{6}[.]png$")
    mask_five_digit_pattern = re.compile(r"^mask\_\d{5}[.]png$")
    mask_six_digit_pattern = re.compile(r"^mask\_\d{6}[.]png$")
    depth_five_digit_counter = 0
    depth_six_digit_counter = 0
    mask_five_digit_counter = 0
    mask_six_digit_counter = 0
    for filename in os.listdir(frames_path):
        if re.findall(depth_five_digit_pattern, filename):
            depth_five_digit_counter += 1
        elif re.findall(depth_six_digit_pattern, filename):
            depth_six_digit_counter += 1
        elif re.findall(mask_five_digit_pattern, filename):
            mask_five_digit_counter += 1
        elif re.findall(mask_six_digit_pattern, filename):
            mask_six_digit_counter += 1
    frame_count = max(depth_five_digit_counter, depth_six_digit_counter)
    filename_format = FrameFilenameFormat.FIVE_DIGIT if depth_five_digit_counter > depth_six_digit_counter \
        else FrameFilenameFormat.SIX_DIGIT
    use_masks = False
    if not turn_mask_checking_off and mask_five_digit_counter > 0 or mask_six_digit_counter > 0:
        use_masks = True
        if (filename_format == FrameFilenameFormat.FIVE_DIGIT and mask_five_digit_counter != depth_five_digit_counter) \
                or (filename_format == FrameFilenameFormat.SIX_DIGIT and
                    mask_six_digit_counter != depth_six_digit_counter):
            print("WARNING: Found some mask files, but could not establish correspondence with depth frames. "
                  "To be matched to depth filenames, mask filenames should use the same numbering format as the "
                  "depth filenames, there should be an equal number of both masks and depth frames, and the numbering"
                  "ranges should correspond.")
            use_masks = False
    return frame_count, filename_format, use_masks


def generate_frame_path_format_string(frame_directory, frame_filename_format):
    if frame_filename_format == FrameFilenameFormat.SIX_DIGIT:
        frame_path_format_string = frame_directory + os.path.sep + "depth_{:0>6d}.png"
        mask_path_format_string = frame_directory + os.path.sep + "mask_{:0>6d}.png"
    else:  # has to be FIVE_DIGIT
        frame_path_format_string = frame_directory + os.path.sep + "depth_{:0>5d}.png"
        mask_path_format_string = frame_directory + os.path.sep + "mask_{:0>5d}.png"

    return frame_path_format_string, mask_path_format_string


def prepare_datasets_for_2d_frame_pair_processing(
        calibration_path=os.path.join(pu.get_reconstruction_data_directory(),
                                      "real_data/snoopy/snoopy_calib.txt"),
        frame_directory=os.path.join(pu.get_reconstruction_data_directory(),
                                     "real_data/snoopy/frames/"),
        output_directory="./output",
        y_range=(214, 400),
        replace_empty_rows=True,
        use_masks=True,
        offset=np.array([-64, -64, 128]),
        field_size=128,

        input_case_file=None,
        case_file_contains_live_index=False,
        case_file_contains_column_header=False,

):
    """

    :param calibration_path: path to calibration file
    :param frame_directory: directory where depth, color, and, potentially, mask images reside, all postfixed with
     frame numbers
    :param output_directory: general output directory
    TODO: make safe_y_range a property of each dataset
    :param y_range: some kind of estimated range of "good" pixel rows that works for every image in the dataset
    :param replace_empty_rows: make sure we don't use rows full of zeros from the depth images
    :param use_masks: whether or not to use masks (if such are included in the frame_directory)
    :param input_case_file: a .csv dataset specifying exactly which frame pairs and rows to use, along with
    focus (image) coordinates for debugging
    (format is TBD, but for now it is: canonical_frame_index, pixel_row_index, focus_x, focus_y)
    :param case_file_contains_live_index: set to True to support case files with live_frame_index as an extra
    second column
    :param case_file_contains_column_header: set to True if the case file's first row lists column names
    :param offset: offset, in voxels, of TSDF fields from camera
    :param field_size: side length of (square) TSDF field in voxels
    :return: a set of frame datasets
    """
    frame_count, frame_filename_format, use_masks = check_frame_count_and_format(frame_directory, not use_masks)
    frame_path_format_string, mask_path_format_string = \
        generate_frame_path_format_string(frame_directory, frame_filename_format)

    if input_case_file:
        frame_row_and_focus_set = np.genfromtxt(input_case_file, delimiter=",", dtype=np.int32)
        # drop column headers
        if case_file_contains_column_header:
            frame_row_and_focus_set = frame_row_and_focus_set[1:]

        # drop live frame indexes
        if case_file_contains_live_index:
            frame_row_and_focus_set = np.concatenate(
                (frame_row_and_focus_set[:, 0].reshape(-1, 1), frame_row_and_focus_set[:, 2].reshape(-1, 1),
                 frame_row_and_focus_set[:, 3:5]), axis=1)
    else:
        frame_set = list(range(0, frame_count - 1, 5))
        pixel_row_set = y_range[0] + ((y_range[1] - y_range[0]) * np.random.rand(len(frame_set))).astype(
            np.int32)
        focus_x = np.zeros((len(frame_set), 1,))
        focus_y = np.zeros((len(frame_set), 1,))
        frame_row_and_focus_set = zip(frame_set, pixel_row_set, focus_x, focus_y)
        if replace_empty_rows:
            new_pixel_row_set = []
            for canonical_frame_index, pixel_row_index, _, _ in frame_row_and_focus_set:
                live_frame_index = canonical_frame_index + 1
                canonical_frame_path = frame_path_format_string.format(canonical_frame_index)
                canonical_mask_path = mask_path_format_string.format(canonical_frame_index)
                live_frame_path = frame_path_format_string.format(live_frame_index)
                live_mask_path = mask_path_format_string.format(live_frame_index)
                while is_image_row_empty(canonical_frame_path, canonical_mask_path, pixel_row_index, use_masks) \
                        or is_image_row_empty(live_frame_path, live_mask_path, pixel_row_index, use_masks):
                    pixel_row_index = y_range[0] + (y_range[1] - y_range[0]) * np.random.rand()
                new_pixel_row_set.append(pixel_row_index)
            frame_row_and_focus_set = zip(frame_set, pixel_row_set, focus_x, focus_y)

    datasets = []

    for canonical_frame_index, pixel_row_index, focus_x, focus_y in frame_row_and_focus_set:
        out_subpath = os.path.join(output_directory, "frames {:0>6d}-{:0>6d} line {:0>3d}"
                                   .format(canonical_frame_index, canonical_frame_index + 1, pixel_row_index))
        canonical_frame_path = frame_path_format_string.format(canonical_frame_index)
        canonical_mask_path = mask_path_format_string.format(canonical_frame_index)
        live_frame_path = frame_path_format_string.format(canonical_frame_index + 1)
        live_mask_path = mask_path_format_string.format(canonical_frame_index + 1)
        if use_masks:
            dataset = dts.MaskedImageBasedFramePairDataset(calibration_path, canonical_frame_path, canonical_mask_path,
                                                           live_frame_path, live_mask_path, pixel_row_index,
                                                           field_size, offset, focus_coordinates=(focus_x, focus_y))
        else:
            dataset = dts.ImageBasedFramePairDataset(calibration_path, canonical_frame_path, live_frame_path,
                                                     pixel_row_index, field_size, offset,
                                                     focus_coordinates=(focus_x, focus_y))
        dataset.out_subpath = out_subpath
        datasets.append(dataset)

    return datasets
