#  ================================================================
#  Created by Fei Shan on 11/07/18.
#  Rigid alignment algorithm implementation based on SDF-2-SDF paper.
#  ================================================================

# common libs
import numpy as np

# local
from rigid_opt.sdf_gradient_field import calculate_gradient_wrt_twist
import utils.printing as printing
from rigid_opt.sdf_2_sdf_visualizer import Sdf2SdfVisualizer
from tsdf import generation as tsdf_gen


class Sdf2SdfOptimizer2d:
    """

    """

    class VerbosityParameters:
        """
        Parameters that controls verbosity to stdout.
        Assumes being used in an "immutable" manner, i.e. just a structure that holds values
        """

        def __init__(self, print_max_warp_update=False, print_iteration_energy=False):
            self.print_max_warp_update = print_max_warp_update
            self.print_iteration_energy = print_iteration_energy
            self.per_iteration_flags = [self.print_max_warp_update,
                                        self.print_iteration_energy]
            self.print_per_iteration_info = any(self.per_iteration_flags)

    def __init__(self,
                 verbosity_parameters=None,
                 visualization_parameters=None
                 ):
        """
        Constructor
        :param verbosity_parameters:
        :param visualization_parameters:
        """

        if verbosity_parameters:
            self.verbosity_parameters = verbosity_parameters
        else:
            self.verbosity_parameters = Sdf2SdfOptimizer2d.VerbosityParameters()

        if visualization_parameters:
            self.visualization_parameters = visualization_parameters
        else:
            self.visualization_parameters = Sdf2SdfVisualizer.Parameters()

        self.visualizer = None

    def optimize(self,
                 data_to_use,
                 voxel_size=0.004,
                 narrow_band_width_voxels=20.,
                 iteration=60,
                 eta=.01
                 ):
        """
        Optimization algorithm
        :param data_to_use:
        :param eta: thickness of surface, used to determine reliability of sdf field
        :param iteration: total number of iterations
        :param voxel_size: voxel side length
        :param narrow_band_width_voxels:
        :return:
        """

        canonical_field = data_to_use.generate_2d_canonical_field(narrow_band_width_voxels=narrow_band_width_voxels,
                                                                  method=tsdf_gen.GenerationMethod.BASIC)
        live_field = data_to_use.generate_2d_live_field(narrow_band_width_voxels=narrow_band_width_voxels,
                                                        method=tsdf_gen.GenerationMethod.BASIC)
        field_size = canonical_field.shape[0]
        offset = data_to_use.offset
        twist = np.zeros((3, 1))

        self.visualizer = Sdf2SdfVisualizer(parameters=self.visualization_parameters, field_size=field_size)
        self.visualizer.generate_pre_optimization_visualizations(canonical_field, live_field)

        for iteration_count in range(iteration):
            matrix_a = np.zeros((3, 3))
            vector_b = np.zeros((3, 1))
            canonical_weight = (canonical_field > -eta).astype(np.int)
            live_field = data_to_use.generate_2d_live_field(narrow_band_width_voxels=narrow_band_width_voxels,
                                                            method=tsdf_gen.GenerationMethod.BASIC,
                                                            twist=np.array([twist[0],
                                                                           [0.],
                                                                           twist[1],
                                                                           [0.],
                                                                           twist[2],
                                                                           [0.]], dtype=np.float32))
            live_weight = (live_field > -eta).astype(np.int)
            live_gradient = calculate_gradient_wrt_twist(live_field, twist, array_offset=offset, voxel_size=voxel_size)

            for i in range(live_field.shape[0]):
                for j in range(live_field.shape[1]):
                    matrix_a += np.dot(live_gradient[i, j][:, None], live_gradient[i, j][None, :])
                    vector_b += (canonical_field[i, j] - live_field[i, j] +
                                 np.dot(live_gradient[i, j][None, :], twist)) * live_gradient[i, j][:, None]

            energy = 0.5 * np.sum((canonical_field * canonical_weight - live_field * live_weight) ** 2)
            if self.verbosity_parameters.print_per_iteration_info:
                print("%s[ITERATION %d COMPLETED]%s" % (printing.BOLD_LIGHT_CYAN, iteration_count, printing.RESET),
                      end="")
                if self.verbosity_parameters.print_iteration_energy:
                    print(" energy: %f" % energy, end="")
                    print("")

            if not np.isfinite(np.linalg.cond(matrix_a)):
                print("%sSINGULAR MATRIX!%s" % (printing.BOLD_YELLOW, printing.RESET))
                continue

            twist_star = np.dot(np.linalg.inv(matrix_a), vector_b)
            twist += .5 * np.subtract(twist_star, twist)

            if self.verbosity_parameters.print_max_warp_update:
                print("optimal twist: %f, %f, %f, twist: %f, %f, %f"
                      % (twist_star[0], twist_star[1], twist_star[2], twist[0], twist[1], twist[2]), end="")
                print("")

            self.visualizer.generate_per_iteration_visualizations(
                data_to_use.generate_2d_live_field(narrow_band_width_voxels=narrow_band_width_voxels,
                                                   method=tsdf_gen.GenerationMethod.BASIC,
                                                   twist=np.array([twist[0],
                                                                  [0.],
                                                                  twist[1],
                                                                  [0.],
                                                                  twist[2],
                                                                  [0.]], dtype=np.float32)))

        self.visualizer.generate_post_optimization_visualizations(canonical_field, live_field)
        del self.visualizer
        return twist
