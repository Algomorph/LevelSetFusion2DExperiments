#  ================================================================
#  Created by Fei Shan on 3/19/19.
#  ================================================================

# standard library
from enum import Enum
# requires Python 3.3+

# local
import rigid_opt.sdf_2_sdf_optimizer2d as sdf2sdfo_py
import rigid_opt.sdf_2_sdf_visualizer as sdf2sdfv_py
# has to be built & installed first (git submodule in cpp folder or http://github/Algomorph/LevelSetFusion-CPP)
import level_set_fusion_optimization as sdf2sdfo_cpp


class ImplementationLanguage(Enum):
    PYTHON = 0
    CPP = 1


class Sdf2SdfOptimizer2dSharedParameters:
    def __init__(self,
                 rate=0.5,
                 maximum_iteration_count=60):
        self.rate = rate
        self.maximum_iteration_count = maximum_iteration_count


def make_common_sdf_2_sdf_optimizer2d_visualization_parameters(out_path="out/sdf_2_sdf"):

    visualization_parameters = sdf2sdfv_py.Sdf2SdfVisualizer.Parameters(
        out_path=out_path,
        show_live_progression=True,
        save_live_progression=True,
        save_initial_fields=True,
        save_final_fields=True,
        save_warp_field_progression=True,
        save_data_gradients=True
    )
    return visualization_parameters


def make_common_sdf_2_sdf_optimizer2d_py_verbosity_parameters():
    verbosity_parameters = sdf2sdfo_py.Sdf2SdfOptimizer2d.VerbosityParameters(
        print_max_warp_update=True,
        print_iteration_energy=True
    )
    return verbosity_parameters


def make_sdf_2_sdf_optimizer2d(implementation_language=ImplementationLanguage.CPP,
                               shared_parameters=Sdf2SdfOptimizer2dSharedParameters(),
                               verbosity_parameters_cpp=sdf2sdfo_cpp.Sdf2SdfOptimizer2d.VerbosityParameters(),
                               verbosity_parameters_py=
                               make_common_sdf_2_sdf_optimizer2d_py_verbosity_parameters(),
                               visualization_parameters_py=
                               make_common_sdf_2_sdf_optimizer2d_visualization_parameters()):
    if implementation_language == ImplementationLanguage.CPP:
        return make_cpp_optimizer(shared_parameters, verbosity_parameters_cpp)
    elif implementation_language == ImplementationLanguage.PYTHON:
        return make_python_optimizer(shared_parameters, verbosity_parameters_py, visualization_parameters_py)
    else:
        raise ValueError("Unsupported ImplementationLanguage: " + str(implementation_language))


def make_python_optimizer(shared_parameters=Sdf2SdfOptimizer2dSharedParameters(),
                          verbosity_parameters=make_common_sdf_2_sdf_optimizer2d_py_verbosity_parameters(),
                          visualization_parameters=make_common_sdf_2_sdf_optimizer2d_visualization_parameters()):
    optimizer = sdf2sdfo_py.Sdf2SdfOptimizer2d(
        verbosity_parameters=verbosity_parameters,
        visualization_parameters=visualization_parameters
    )
    return optimizer


def make_cpp_optimizer(shared_parameters=Sdf2SdfOptimizer2dSharedParameters(),
                       verbosity_parameters=sdf2sdfo_cpp.Sdf2SdfOptimizer2d.VerbosityParameters()):
    optimizer = sdf2sdfo_cpp.Sdf2SdfOptimizer2d(
        rate=shared_parameters.rate,
        maximum_iteration_count=shared_parameters.maximum_iteration_count,
        verbosity_parameters=verbosity_parameters
    )
    return optimizer
