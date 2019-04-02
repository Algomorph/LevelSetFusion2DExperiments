# import unittest
from unittest import TestCase
import numpy as np
import math
from calib.camera import DepthCamera
from tsdf import generation as tsdf_gen
from math_utils.transformation import twist_vector_to_matrix3d
import os.path
import cv2


class MyTestCase(TestCase):

    @staticmethod
    def image_load_helper(filename):
        path = os.path.join("tests/test_data", filename)
        if not os.path.exists(path):
            path = os.path.join("test_data", filename)
        depth_image = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        max_depth = np.iinfo(np.uint16).max
        depth_image[depth_image == 0] = max_depth
        return depth_image

    def test_sdf_generation01(self):
        depth_image = np.full((3, 3), math.inf)
        image_pixel_row = 1
        offset = np.array([-1, -1, 1])
        field_size = 3
        narrow_band_width_voxels = 1

        intrinsic_matrix = np.array([[1, 0, 1],  # FX = 1 CX = 1
                                     [0, 1, 1],  # FY = 1 CY = 1
                                     [0, 0, 1]], dtype=np.float32)

        depth_camera = DepthCamera(intrinsics=DepthCamera.Intrinsics(resolution=(3, 3),
                                                                     intrinsic_matrix=intrinsic_matrix),
                                   depth_unit_ratio=1)

        expected_field = np.ones((3, 3))
        field = tsdf_gen.generate_2d_tsdf_field_from_depth_image(depth_image, depth_camera, image_pixel_row,
                                                                 field_size=field_size,
                                                                 default_value=-999,
                                                                 voxel_size=1,
                                                                 array_offset=offset,
                                                                 narrow_band_width_voxels=narrow_band_width_voxels)
        self.assertTrue(np.allclose(expected_field, field))

    def test_sdf_generation02(self):
        depth_image = np.zeros((3, 3))
        image_pixel_row = 1
        offset = np.array([-1, -1, 1])
        field_size = 3
        narrow_band_width_voxels = 1

        intrinsic_matrix = np.array([[1, 0, 1],  # FX = 1 CX = 1
                                     [0, 1, 1],  # FY = 1 CY = 1
                                     [0, 0, 1]], dtype=np.float32)

        depth_camera = DepthCamera(intrinsics=DepthCamera.Intrinsics(resolution=(3, 3),
                                                                     intrinsic_matrix=intrinsic_matrix),
                                   depth_unit_ratio=1)

        expected_field = np.full((3, 3), -999)
        field = tsdf_gen.generate_2d_tsdf_field_from_depth_image(depth_image, depth_camera, image_pixel_row,
                                                                 field_size=field_size,
                                                                 default_value=-999,
                                                                 voxel_size=1,
                                                                 array_offset=offset,
                                                                 narrow_band_width_voxels=narrow_band_width_voxels)
        self.assertTrue(np.allclose(expected_field, field))

    def test_sdf_generation03(self):
        depth_image = np.zeros((3, 3))
        image_pixel_row = 1
        offset = np.array([-1, -1, -1])
        field_size = 3
        narrow_band_width_voxels = 1

        intrinsic_matrix = np.array([[1, 0, 1],  # FX = 1 CX = 1
                                     [0, 1, 1],  # FY = 1 CY = 1
                                     [0, 0, 1]], dtype=np.float32)

        depth_camera = DepthCamera(intrinsics=DepthCamera.Intrinsics(resolution=(3, 3),
                                                                     intrinsic_matrix=intrinsic_matrix),
                                   depth_unit_ratio=1)

        expected_field = np.full((3, 3), -999)
        field = tsdf_gen.generate_2d_tsdf_field_from_depth_image(depth_image, depth_camera, image_pixel_row,
                                                                 field_size=field_size,
                                                                 default_value=-999,
                                                                 voxel_size=1,
                                                                 array_offset=offset,
                                                                 narrow_band_width_voxels=narrow_band_width_voxels, )
        self.assertTrue(np.allclose(expected_field, field))

    def test_sdf_generation04(self):
        depth_image = np.ones((3, 3))
        image_pixel_row = 1
        offset = np.array([-1, -1, 1])
        field_size = 3
        narrow_band_width_voxels = 1

        intrinsic_matrix = np.array([[1, 0, 1],  # FX = 1 CX = 1
                                     [0, 1, 1],  # FY = 1 CY = 1
                                     [0, 0, 1]], dtype=np.float32)

        depth_camera = DepthCamera(intrinsics=DepthCamera.Intrinsics(resolution=(3, 3),
                                                                     intrinsic_matrix=intrinsic_matrix),
                                   depth_unit_ratio=1)

        expected_field = np.array([[0, 0, 0],
                                   [-1, -1, -1],
                                   [-1, -1, -1]])
        field = tsdf_gen.generate_2d_tsdf_field_from_depth_image(depth_image, depth_camera, image_pixel_row,
                                                                 field_size=field_size,
                                                                 default_value=-999,
                                                                 voxel_size=1,
                                                                 array_offset=offset,
                                                                 narrow_band_width_voxels=narrow_band_width_voxels)
        self.assertTrue(np.allclose(expected_field, field))

    def test_sdf_generation05(self):
        depth_image = np.ones((3, 3))
        image_pixel_row = 1
        offset = np.array([-1, -1, 1])
        field_size = 3
        narrow_band_width_voxels = 1
        twist3d = np.zeros((6, 1))
        twist_matrix3d = twist_vector_to_matrix3d(twist3d)

        intrinsic_matrix = np.array([[1, 0, 1],  # FX = 1 CX = 1
                                     [0, 1, 1],  # FY = 1 CY = 1
                                     [0, 0, 1]], dtype=np.float32)

        depth_camera = DepthCamera(intrinsics=DepthCamera.Intrinsics(resolution=(3, 3),
                                                                     intrinsic_matrix=intrinsic_matrix),
                                   depth_unit_ratio=1)

        expected_field = np.array([[0, 0, 0],
                                   [-1, -1, -1],
                                   [-1, -1, -1]])
        field = tsdf_gen.generate_2d_tsdf_field_from_depth_image(depth_image, depth_camera, image_pixel_row,
                                                                 camera_extrinsic_matrix=twist_matrix3d,
                                                                 field_size=field_size,
                                                                 default_value=-999,
                                                                 voxel_size=1,
                                                                 array_offset=offset,
                                                                 narrow_band_width_voxels=narrow_band_width_voxels)
        self.assertTrue(np.allclose(expected_field, field))

    def test_sdf_generation06(self):
        depth_image = np.ones((3, 3))
        image_pixel_row = 1
        offset = np.array([-1, -1, 1])
        field_size = 3
        narrow_band_width_voxels = 1
        twist3d = np.zeros((6, 1))
        twist3d[1] = 10000000
        twist_matrix3d = twist_vector_to_matrix3d(twist3d)
        intrinsic_matrix = np.array([[1, 0, 1],  # FX = 1 CX = 1
                                     [0, 1, 1],  # FY = 1 CY = 1
                                     [0, 0, 1]], dtype=np.float32)

        depth_camera = DepthCamera(intrinsics=DepthCamera.Intrinsics(resolution=(3, 3),
                                                                     intrinsic_matrix=intrinsic_matrix),
                                   depth_unit_ratio=1)

        expected_field = np.array([[0, 0, 0],
                                   [-1, -1, -1],
                                   [-1, -1, -1]])
        field = tsdf_gen.generate_2d_tsdf_field_from_depth_image(depth_image, depth_camera, image_pixel_row,
                                                                 camera_extrinsic_matrix=twist_matrix3d,
                                                                 field_size=field_size,
                                                                 default_value=-999,
                                                                 voxel_size=1,
                                                                 array_offset=offset,
                                                                 narrow_band_width_voxels=narrow_band_width_voxels)
        self.assertTrue(np.allclose(expected_field, field))

    def test_sdf_generation07(self):
        depth_image = np.ones((3, 3))
        image_pixel_row = 1
        offset = np.array([-1, -1, 1])
        field_size = 3
        narrow_band_width_voxels = 1
        twist3d = np.zeros((6, 1))
        twist3d[0] = 1
        twist_matrix3d = twist_vector_to_matrix3d(twist3d)
        intrinsic_matrix = np.array([[1, 0, 1],  # FX = 1 CX = 1
                                     [0, 1, 1],  # FY = 1 CY = 1
                                     [0, 0, 1]], dtype=np.float32)

        depth_camera = DepthCamera(intrinsics=DepthCamera.Intrinsics(resolution=(3, 3),
                                                                     intrinsic_matrix=intrinsic_matrix),
                                   depth_unit_ratio=1)

        expected_field = np.array([[0, 0, -999],
                                   [-1, -1, -1],
                                   [-1, -1, -1]])
        field = tsdf_gen.generate_2d_tsdf_field_from_depth_image(depth_image, depth_camera, image_pixel_row,
                                                                 camera_extrinsic_matrix=twist_matrix3d,
                                                                 field_size=field_size,
                                                                 default_value=-999,
                                                                 voxel_size=1,
                                                                 array_offset=offset,
                                                                 narrow_band_width_voxels=narrow_band_width_voxels)
        self.assertTrue(np.allclose(expected_field, field))

    def test_sdf_generation08(self):
        depth_image = np.ones((3, 3))
        image_pixel_row = 1
        offset = np.array([-1, -1, 1])
        field_size = 3
        narrow_band_width_voxels = 1
        twist3d = np.zeros((6, 1))
        twist3d[0] = 2
        twist_matrix3d = twist_vector_to_matrix3d(twist3d)
        intrinsic_matrix = np.array([[1, 0, 1],  # FX = 1 CX = 1
                                     [0, 1, 1],  # FY = 1 CY = 1
                                     [0, 0, 1]], dtype=np.float32)

        depth_camera = DepthCamera(intrinsics=DepthCamera.Intrinsics(resolution=(3, 3),
                                                                     intrinsic_matrix=intrinsic_matrix),
                                   depth_unit_ratio=1)

        expected_field = np.array([[0, -999, -999],
                                   [-1, -1, -999],
                                   [-1, -1, -1]])
        field = tsdf_gen.generate_2d_tsdf_field_from_depth_image(depth_image, depth_camera, image_pixel_row,
                                                                 camera_extrinsic_matrix=twist_matrix3d,
                                                                 field_size=field_size,
                                                                 default_value=-999,
                                                                 voxel_size=1,
                                                                 array_offset=offset,
                                                                 narrow_band_width_voxels=narrow_band_width_voxels)
        self.assertTrue(np.allclose(expected_field, field))

    def test_sdf_generation09(self):
        depth_image = np.ones((3, 3))
        image_pixel_row = 1
        offset = np.array([-1, -1, 1])
        field_size = 3
        narrow_band_width_voxels = 1
        twist3d = np.zeros((6, 1))
        twist3d[2] = 1000000
        twist_matrix3d = twist_vector_to_matrix3d(twist3d)
        intrinsic_matrix = np.array([[1, 0, 1],  # FX = 1 CX = 1
                                     [0, 1, 1],  # FY = 1 CY = 1
                                     [0, 0, 1]], dtype=np.float32)

        depth_camera = DepthCamera(intrinsics=DepthCamera.Intrinsics(resolution=(3, 3),
                                                                     intrinsic_matrix=intrinsic_matrix),
                                   depth_unit_ratio=1)

        expected_field = np.array([[-1, -1, -1],
                                   [-1, -1, -1],
                                   [-1, -1, -1]])
        field = tsdf_gen.generate_2d_tsdf_field_from_depth_image(depth_image, depth_camera, image_pixel_row,
                                                                 camera_extrinsic_matrix=twist_matrix3d,
                                                                 field_size=field_size,
                                                                 default_value=-999,
                                                                 voxel_size=1,
                                                                 array_offset=offset,
                                                                 narrow_band_width_voxels=narrow_band_width_voxels)
        self.assertTrue(np.allclose(expected_field, field))

    def test_sdf_generation10(self):
        depth_image = np.ones((3, 3))
        image_pixel_row = 1
        offset = np.array([-1, -1, 1])
        field_size = 3
        narrow_band_width_voxels = 1
        twist3d = np.zeros((6, 1))
        twist3d[2] = -1
        twist_matrix3d = twist_vector_to_matrix3d(twist3d)
        intrinsic_matrix = np.array([[1, 0, 1],  # FX = 1 CX = 1
                                     [0, 1, 1],  # FY = 1 CY = 1
                                     [0, 0, 1]], dtype=np.float32)

        depth_camera = DepthCamera(intrinsics=DepthCamera.Intrinsics(resolution=(3, 3),
                                                                     intrinsic_matrix=intrinsic_matrix),
                                   depth_unit_ratio=1)

        expected_field = np.array([[-999, -999, -999],
                                   [0, 0, 0],
                                   [-1, -1, -1]])
        field = tsdf_gen.generate_2d_tsdf_field_from_depth_image(depth_image, depth_camera, image_pixel_row,
                                                                 camera_extrinsic_matrix=twist_matrix3d,
                                                                 field_size=field_size,
                                                                 default_value=-999,
                                                                 voxel_size=1,
                                                                 array_offset=offset,
                                                                 narrow_band_width_voxels=narrow_band_width_voxels)
        self.assertTrue(np.allclose(expected_field, field))

    # def test_sdf_generation11(self):
    #     filename = "zigzag2_depth_00108.png"
    #     depth_image = self.image_load_helper(filename)
    #     image_pixel_row = 200
    #     offset = np.array([-8, -8, 144])
    #     field_size = 16
    #     narrow_band_width_voxels = 20
    #     camera_intrinsic_matrix = np.array([[700., 0., 320.],
    #                                         [0., 700., 240.],
    #                                         [0., 0., 1.]])
    #
    #     depth_camera = DepthCamera(intrinsics=DepthCamera.Intrinsics((640, 480), intrinsic_matrix=camera_intrinsic_matrix),
    #                                depth_unit_ratio=0.001)
    #
    #     field = tsdf_gen.generate_2d_tsdf_field_from_depth_image(depth_image, depth_camera, image_pixel_row,
    #                                                              camera_extrinsic_matrix=np.eye(4),
    #                                                              field_size=field_size,
    #                                                              default_value=-999,
    #                                                              voxel_size=0.004,
    #                                                              array_offset=offset,
    #                                                              narrow_band_width_voxels=narrow_band_width_voxels)
    #     print(np.array2string(field, precision=6, separator=', ', formatter={'float': lambda x: "%.8f" % x + 'f'}))