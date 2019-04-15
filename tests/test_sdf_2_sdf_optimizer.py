from unittest import TestCase
import numpy as np
import cv2
from calib.camera import DepthCamera
from rigid_opt import sdf_2_sdf_visualizer as sdf2sdfv, sdf_2_sdf_optimizer2d as sdf2sdfo_py
from rigid_opt.sdf_generation import ImageBasedSingleFrameDataset
import utils.sampling as sampling
import utils.path
import experiment.build_sdf_2_sdf_optimizer_helper as build_opt

# C++ extension
import level_set_fusion_optimization as sdf2sdfo_cpp


class MyTestCase(TestCase):

    def test_sdf_2_sdf_optimizer01(self):
        canonical_frame_path = utils.path.get_test_data_path("test_data/depth_000000.exr")
        live_frame_path = utils.path.get_test_data_path("test_data/depth_000003.exr")

        image_pixel_row = 240

        intrinsic_matrix = np.array([[570.3999633789062, 0, 320],  # FX = 570.3999633789062 CX = 320.0
                                      [0, 570.3999633789062, 240],  # FY = 570.3999633789062 CY = 240.0
                                      [0, 0, 1]], dtype=np.float32)
        camera = DepthCamera(intrinsics=DepthCamera.Intrinsics(resolution=(480, 640),
                                                               intrinsic_matrix=intrinsic_matrix))
        field_size = 32
        offset = np.array([[-16], [-16], [93.4375]])

        data_to_use = ImageBasedSingleFrameDataset(
            canonical_frame_path,  # dataset from original sdf2sdf paper, reference frame
            live_frame_path,  # dataset from original sdf2sdf paper, current frame
            image_pixel_row, field_size, offset, camera
        )

        # depth_interpolation_method = tsdf.DepthInterpolationMethod.NONE
        out_path = "output/test_rigid_out"
        sampling.set_focus_coordinates(0, 0)
        narrow_band_width_voxels = 2.
        iteration = 40
        optimizer = sdf2sdfo_py.Sdf2SdfOptimizer2d(
            verbosity_parameters=sdf2sdfo_py.Sdf2SdfOptimizer2d.VerbosityParameters(
                print_max_warp_update=False,
                print_iteration_energy=False
            ),
            visualization_parameters=sdf2sdfv.Sdf2SdfVisualizer.Parameters(
                out_path=out_path,
                save_initial_fields=False,
                save_final_fields=False,
                save_live_progression=False
            )
        )
        expected_twist = np.array([[-0.08019894],
                                   [0.00629552],
                                   [0.16065715]])
        twist = optimizer.optimize(data_to_use, narrow_band_width_voxels=narrow_band_width_voxels, iteration=iteration)

        self.assertTrue(np.allclose(expected_twist, twist, atol=10e-6))

    def test_operation_same_cpp_to_py(self):
        canonical_frame_path = utils.path.get_test_data_path("test_data/depth_000000.exr")
        live_frame_path = utils.path.get_test_data_path("test_data/depth_000003.exr")
        image_pixel_row = 240
        intrinsic_matrix = np.array([[570.3999633789062, 0, 320],  # FX = 570.3999633789062 CX = 320.0
                                     [0, 570.3999633789062, 240],  # FY = 570.3999633789062 CY = 240.0
                                     [0, 0, 1]], dtype=np.float32)
        camera = DepthCamera(intrinsics=DepthCamera.Intrinsics(resolution=(480, 640),
                                                               intrinsic_matrix=intrinsic_matrix))
        field_size = 32
        offset = np.array([[-16], [-16], [93]], dtype=np.int32)
        voxel_size = 0.004
        narrow_band_width_voxels = 2
        eta = 0.01  # thickness, used to calculate sdf weight.

        shared_parameters = build_opt.Sdf2SdfOptimizer2dSharedParameters()
        shared_parameters.rate = 0.5
        shared_parameters.maximum_iteration_count = 8

        # For verbose output from py version:
        verbosity_parameters_py = sdf2sdfo_py.Sdf2SdfOptimizer2d.VerbosityParameters(True, True)
        verbosity_parameters_cpp = sdf2sdfo_cpp.Sdf2SdfOptimizer2d.VerbosityParameters(True, True)
        visualization_parameters_py = sdf2sdfv.Sdf2SdfVisualizer.Parameters()
        visualization_parameters_py.out_path = "out"

        # For c++ TSDF generator
        tsdf_generation_parameters = sdf2sdfo_cpp.Sdf2SdfOptimizer2d.TSDFGenerationParameters(
            depth_unit_ratio=0.001,  # mm to meter
            camera_intrinsic_matrix=intrinsic_matrix,
            camera_pose=np.eye(4, dtype=np.float32),
            array_offset=offset,
            field_size=field_size,
            voxel_size=voxel_size,
            narrow_band_width_voxels=narrow_band_width_voxels
        )

        # Read image for c++ optimizer, identical to python, which is done inside ImageBasedSingleFrameDataset class.
        canonical_depth_image = cv2.imread(canonical_frame_path, cv2.IMREAD_UNCHANGED)
        canonical_depth_image = cv2.cvtColor(canonical_depth_image, cv2.COLOR_BGR2GRAY)
        canonical_depth_image = canonical_depth_image.astype(np.uint16)  # mm
        canonical_depth_image[canonical_depth_image == 0] = np.iinfo(np.uint16).max

        live_depth_image = cv2.imread(live_frame_path, cv2.IMREAD_UNCHANGED)
        live_depth_image = cv2.cvtColor(live_depth_image, cv2.COLOR_BGR2GRAY)
        live_depth_image = live_depth_image.astype(np.uint16)  # mm
        live_depth_image[live_depth_image == 0] = np.iinfo(np.uint16).max

        optimizer_cpp = build_opt.make_sdf_2_sdf_optimizer2d(
            implementation_language=build_opt.ImplementationLanguage.CPP,
            shared_parameters=shared_parameters,
            verbosity_parameters_cpp=verbosity_parameters_cpp,
            verbosity_parameters_py=verbosity_parameters_py,
            visualization_parameters_py=visualization_parameters_py)

        twist_cpp = optimizer_cpp.optimize(image_y_coordinate=image_pixel_row,
                                           canonical_depth_image=canonical_depth_image,
                                           live_depth_image=live_depth_image,
                                           tsdf_generation_parameters=tsdf_generation_parameters,
                                           eta=eta)

        # For python optimizer
        data_to_use = ImageBasedSingleFrameDataset( # for python
            canonical_frame_path,  # dataset from original sdf2sdf paper, reference frame
            live_frame_path,  # dataset from original sdf2sdf paper, current frame
            image_pixel_row, field_size, offset, camera
        )

        optimizer_py = build_opt.make_sdf_2_sdf_optimizer2d(
            implementation_language=build_opt.ImplementationLanguage.PYTHON,
            shared_parameters=shared_parameters,
            verbosity_parameters_cpp=verbosity_parameters_cpp,
            verbosity_parameters_py=verbosity_parameters_py,
            visualization_parameters_py=visualization_parameters_py)

        twist_py = optimizer_py.optimize(data_to_use,
                                         voxel_size=voxel_size,
                                         narrow_band_width_voxels=narrow_band_width_voxels,
                                         iteration=shared_parameters.maximum_iteration_count,
                                         eta=eta)

        self.assertTrue(np.allclose(twist_cpp, twist_py, atol=1e-5))



