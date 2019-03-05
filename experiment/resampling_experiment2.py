#  ================================================================
#  Created by Gregory Kramida on 1/22/19.
#  Copyright (c) 2019 Gregory Kramida
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

# stdlib
import sys
import os.path

# libs
import numpy as np
import cv2
import matplotlib.pyplot as plt

# local
import tsdf.ewa as ewa
import tsdf.generation as gen
import experiment.dataset as data
import utils.visualization as viz

# =========

from calib.camerarig import DepthCameraRig

EXIT_CODE_SUCCESS = 0
EXIT_CODE_FAILURE = 1


def main():
    # array_offset = np.array([-256, -256, 480], dtype=np.int32) # zigzag 64
    array_offset = np.array([-256, -256, 0], dtype=np.int32)  # zigzag2 108
    field_size = np.array([512, 512, 512], dtype=np.int32)
    voxel_size = 0.004
    rig = DepthCameraRig.from_infinitam_format(
        "/media/algomorph/Data/Reconstruction/synthetic_data/zigzag/inf_calib.txt")
    depth_camera = rig.depth_camera
    depth_interpolation_method = gen.GenerationMethod.EWA_IMAGE
    # depth_image0 = cv2.imread(
    #     "/media/algomorph/Data/Reconstruction/synthetic_data/zigzag/input/depth_00064.png",
    #     cv2.IMREAD_UNCHANGED)
    depth_image0 = cv2.imread(
        # "/media/algomorph/Data/Reconstruction/synthetic_data/zigzag2/input/depth_00000.png",
        "/media/algomorph/Data/Reconstruction/synthetic_data/zigzag2/input/depth_00108.png",
        cv2.IMREAD_UNCHANGED)
    max_depth = np.iinfo(np.uint16).max
    depth_image0[depth_image0 == 0] = max_depth
    field = \
        ewa.generate_tsdf_3d_ewa_image_cpp(depth_image0,
                                           depth_camera,
                                           field_shape=field_size,
                                           array_offset=array_offset,
                                           voxel_size=voxel_size,
                                           narrow_band_width_voxels=20)
    viz_image = ewa.generate_tsdf_3d_ewa_image_visualization_cpp(depth_image=depth_image0,
                                                                 camera=depth_camera,
                                                                 field=field,
                                                                 voxel_size=voxel_size,
                                                                 array_offset=array_offset)
    # print(viz_image.shape, viz_image.dtype)
    # resized = cv2.resize(viz_image, (2400, 3200))
    cv2.imwrite("../output/ewa_sampling_viz.png", viz_image)

    return EXIT_CODE_SUCCESS


if __name__ == "__main__":
    sys.exit(main())
