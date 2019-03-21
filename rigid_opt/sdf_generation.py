#  ================================================================
#  Created by Fei Shan on 01/31/19.
#  sdf generation, separate live field and canonical field generation, allow applying twist to live pc
#  ================================================================

# common libs
import numpy as np
import cv2

# local
from tsdf import generation as tsdf_gen
from math_utils.transformation import twist_vector_to_matrix3d


class ImageBasedSingleFrameDataset:
    def __init__(self, first_frame_path, second_frame_path, image_pixel_row, field_size, offset, camera):
        self.first_frame_path = first_frame_path
        self.second_frame_path = second_frame_path
        self.image_pixel_row = image_pixel_row
        self.field_size = field_size
        self.offset = offset
        self.depth_camera = camera

    def generate_2d_sdf_fields(self, narrow_band_width_voxels=20., method=tsdf_gen.GenerationMethod.BASIC):
        canonical_field = self.generate_2d_canonical_field(narrow_band_width_voxels=narrow_band_width_voxels,
                                                           method=method)
        live_field = self.generate_2d_live_field(narrow_band_width_voxels=narrow_band_width_voxels,
                                                 method=method)
        return live_field, canonical_field

    def generate_2d_canonical_field(self, narrow_band_width_voxels=20., method=tsdf_gen.GenerationMethod.BASIC):
        depth_image0 = cv2.imread(self.first_frame_path, -1)
        depth_image0 = cv2.cvtColor(depth_image0, cv2.COLOR_BGR2GRAY)
        depth_image0 = depth_image0.astype(float)  # cm

        canonical_field = \
            tsdf_gen.generate_2d_tsdf_field_from_depth_image(depth_image0, self.depth_camera, self.image_pixel_row,
                                                             field_size=self.field_size,
                                                             array_offset=self.offset,
                                                             narrow_band_width_voxels=narrow_band_width_voxels,
                                                             generation_method=method)
        # print(str(depth_image0[240]))
        return canonical_field

    def generate_2d_live_field(self, method=tsdf_gen.GenerationMethod.BASIC,
                               narrow_band_width_voxels=20.,
                               twist=np.zeros((6, 1))):
        depth_image1 = cv2.imread(self.second_frame_path, -1)
        depth_image1 = cv2.cvtColor(depth_image1, cv2.COLOR_BGR2GRAY)
        depth_image1 = depth_image1.astype(float)  # cm

        twist_matrix = twist_vector_to_matrix3d(twist)

        live_field = \
            tsdf_gen.generate_2d_tsdf_field_from_depth_image(depth_image1, self.depth_camera, self.image_pixel_row,
                                                             camera_extrinsic_matrix=twist_matrix,
                                                             field_size=self.field_size,
                                                             array_offset=self.offset,
                                                             narrow_band_width_voxels=narrow_band_width_voxels,
                                                             generation_method=method)
        # print(str(depth_image1[240]))
        return live_field


class ArrayBasedSingleFrameDataset:
    def __init__(self, depth_image0, depth_image1, image_pixel_row, field_size, offset, camera):
        self.depth_image0 = depth_image0
        self.depth_image1 = depth_image1
        self.image_pixel_row = image_pixel_row
        self.field_size = field_size
        self.offset = offset
        self.depth_camera = camera

    def generate_2d_sdf_fields(self, narrow_band_width_voxels=20., method=tsdf_gen.GenerationMethod.BASIC):
        canonical_field = self.generate_2d_canonical_field(narrow_band_width_voxels=narrow_band_width_voxels,
                                                           method=method)
        live_field = self.generate_2d_live_field(narrow_band_width_voxels=narrow_band_width_voxels,
                                                 method=method)
        return live_field, canonical_field

    def generate_2d_canonical_field(self, narrow_band_width_voxels=20., method=tsdf_gen.GenerationMethod.BASIC):
        canonical_field = \
            tsdf_gen.generate_2d_tsdf_field_from_depth_image(self.depth_image0, self.depth_camera, self.image_pixel_row,
                                                             field_size=self.field_size,
                                                             array_offset=self.offset,
                                                             narrow_band_width_voxels=narrow_band_width_voxels,
                                                             generation_method=method)
        return canonical_field

    def generate_2d_live_field(self, method=tsdf_gen.GenerationMethod.BASIC,
                               narrow_band_width_voxels=20.,
                               twist=np.zeros((6, 1))):
        twist_matrix = twist_vector_to_matrix3d(twist)

        live_field = \
            tsdf_gen.generate_2d_tsdf_field_from_depth_image(self.depth_image1, self.depth_camera, self.image_pixel_row,
                                                             camera_extrinsic_matrix=twist_matrix,
                                                             field_size=self.field_size,
                                                             array_offset=self.offset,
                                                             narrow_band_width_voxels=narrow_band_width_voxels,
                                                             generation_method=method)
        return live_field
