#  ================================================================
#  Created by Gregory Kramida on 1/31/19.
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

from unittest import TestCase
import numpy as np
import calib.camera as cam
import tests.ewa_test_data as data
import tsdf.ewa as ewa
import os.path
import cv2


class TsdfTest(TestCase):
    def test_2D_ewa_tsdf_generation1(self):
        depth_image = np.zeros((3, 640), dtype=np.uint16)
        depth_image[:] = np.iinfo(np.uint16).max
        depth_image_region = np.array([[3233, 3246, 3243, 3256, 3253, 3268, 3263, 3279, 3272, 3289, 3282,
                                        3299, 3291, 3308, 3301, 3317, 3310, 3326],
                                       [3233, 3246, 3243, 3256, 3253, 3268, 3263, 3279, 3272, 3289, 3282,
                                        3299, 3291, 3308, 3301, 3317, 3310, 3326],
                                       [3233, 3246, 3243, 3256, 3253, 3268, 3263, 3279, 3272, 3289, 3282,
                                        3299, 3291, 3308, 3301, 3317, 3310, 3326]], dtype=np.uint16)
        depth_image[:, 399:417] = depth_image_region
        camera_intrisic_matrix = np.array([[700., 0., 320.],
                                           [0., 700., 240.],
                                           [0., 0., 1.]])
        camera = cam.DepthCamera(intrinsics=cam.Camera.Intrinsics((640, 3), intrinsic_matrix=camera_intrisic_matrix),
                                 depth_unit_ratio=0.001)
        field = \
            ewa.generate_2d_tsdf_field_from_depth_image_ewa(depth_image, camera, 1,
                                                            field_size=16,
                                                            array_offset=np.array([94, -256, 804]),
                                                            voxel_size=0.004)
        self.assertTrue(np.allclose(field, data.out_sdf_field1, atol=2e-5))

        field2 = ewa.generate_2d_tsdf_field_from_depth_image_ewa_cpp(depth_image, camera, 1,
                                                                     field_size=16,
                                                                     array_offset=np.array([94, -256, 804],
                                                                                           dtype=np.int32),
                                                                     voxel_size=0.004)
        self.assertTrue(np.allclose(field2, data.out_sdf_field1, atol=1e-6))

    def image_load_helper(self, filename):
        path = os.path.join("tests/test_data", filename)
        if not os.path.exists(path):
            path = os.path.join("test_data", filename)
        depth_image = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        max_depth = np.iinfo(np.uint16).max
        depth_image[depth_image == 0] = max_depth
        return depth_image

    def test_2d_ewa_tsdf_generation2(self):
        filename = "zigzag2_depth_00108.png"
        depth_image = self.image_load_helper(filename)
        camera_intrisic_matrix = np.array([[700., 0., 320.],
                                           [0., 700., 240.],
                                           [0., 0., 1.]])
        camera = cam.DepthCamera(intrinsics=cam.Camera.Intrinsics((640, 480), intrinsic_matrix=camera_intrisic_matrix),
                                 depth_unit_ratio=0.001)

        field2 = ewa.generate_2d_tsdf_field_from_depth_image_ewa_cpp(depth_image, camera, 200,
                                                                     field_size=512,
                                                                     array_offset=np.array([-256, -256, 0]),
                                                                     voxel_size=0.004)
        chunk = field2[103:119, 210:226].copy()
        self.assertTrue(np.allclose(chunk, data.out_sdf_chunk1))

        field = \
            ewa.generate_2d_tsdf_field_from_depth_image_ewa(depth_image, camera, 200,
                                                            field_size=512,
                                                            array_offset=np.array([-256, -256, 0]),
                                                            voxel_size=0.004)
        chunk = field[103:119, 210:226].copy()
        self.assertTrue(np.allclose(chunk, data.out_sdf_chunk1))
