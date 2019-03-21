#  ================================================================
#  Created by Fei Shan on 01/23/19.
#  Rigid alignment algorithm implementation based on SDF-2-SDF paper visualization.
#  ================================================================

# stdlib
import os.path
import os
# libraries
import cv2
# local
import utils.visualization as viz


class Sdf2SdfVisualizer:

    class Parameters:
        def __init__(self, out_path="output/sdf_2_sdf_optimizer/", view_scaling_factor=8,
                     show_live_progression=False,
                     save_live_progression=False,
                     save_initial_fields=False,
                     save_final_fields=False,
                     save_warp_field_progression=False,
                     save_data_gradients=False):
            self.out_path = out_path
            self.view_scaling_factor = view_scaling_factor
            self.show_live_progress = show_live_progression

            self.save_live_field_progression = save_live_progression
            self.save_initial_fields = save_initial_fields
            self.save_final_fields = save_final_fields
            self.save_warp_field_progression = save_warp_field_progression
            self.save_data_gradients = save_data_gradients
            self.using_output_folder = self.save_final_fields or \
                                       self.save_initial_fields or \
                                       self.save_live_field_progression or \
                                       self.save_warp_field_progression or \
                                       self.save_data_gradients

    def __init__(self, parameters=None, field_size=128, level_count=4):
        self.field_size = field_size
        self.parameters = parameters
        self.level_count = level_count
        if not parameters:
            self.parameters = Sdf2SdfVisualizer.Parameters()
        # initialize video-writers
        self.live_progression_writer = None
        self.warp_video_writer2D = None
        self.data_gradient_video_writer2D = None

        if self.parameters.using_output_folder:
            if not os.path.exists(self.parameters.out_path):
                os.makedirs(self.parameters.out_path)

        if self.parameters.save_live_field_progression:
            self.live_progression_writer = cv2.VideoWriter(
                os.path.join(self.parameters.out_path, 'live_field_evolution_2D.mkv'),
                cv2.VideoWriter_fourcc('X', '2', '6', '4'), 10,
                (field_size * self.parameters.view_scaling_factor, field_size * self.parameters.view_scaling_factor),
                isColor=False)
        if self.parameters.save_warp_field_progression:
            self.warp_video_writer2D = cv2.VideoWriter(
                os.path.join(self.parameters.out_path, 'warp_2D_quiverplot.mkv'),
                cv2.VideoWriter_fourcc('X', '2', '6', '4'), 10, (1920, 1200), isColor=True)

        if self.parameters.save_data_gradients:
            self.data_gradient_video_writer2D = cv2.VideoWriter(
                os.path.join(self.parameters.out_path, 'data_gradient_2D_quiverplot.mkv'),
                cv2.VideoWriter_fourcc('X', '2', '6', '4'), 10, (1920, 1200), isColor=True)

    def generate_pre_optimization_visualizations(self, canonical_field, live_field):
        if self.parameters.save_initial_fields:
            viz.save_initial_fields(canonical_field, live_field, self.parameters.out_path,
                                    self.parameters.view_scaling_factor)

    def generate_post_optimization_visualizations(self, canonical_field, live_field):
        if self.parameters.save_final_fields:
            viz.save_final_fields(canonical_field, live_field, self.parameters.out_path,
                                  self.parameters.view_scaling_factor)

    def generate_per_iteration_visualizations(self, live_field):
        if self.parameters.save_live_field_progression:
            live_field_out = viz.sdf_field_to_image(live_field, self.parameters.view_scaling_factor)
            self.live_progression_writer.write(live_field_out)

        if self.parameters.save_warp_field_progression:
            upscaled_warp_field = warp_field.repeat(level_scaling, axis=0).repeat(level_scaling, axis=1)
            self.warp_video_writer2D.write(
                viz.make_vector_field_plot(upscaled_warp_field, scale=1.0, iteration_number=iteration_number,
                                           vectors_name="Warp vectors"))
        if self.parameters.save_data_gradients:
            upscaled_data_gradient = data_gradient.repeat(level_scaling, axis=0).repeat(level_scaling, axis=1)
            self.data_gradient_video_writer2D.write(
                viz.make_vector_field_plot(upscaled_data_gradient, scale=10.0, iteration_number=iteration_number,
                                           vectors_name="Data gradient (10X magnitude)"))

    def __del__(self):
        if self.live_progression_writer:
            self.live_progression_writer.release()
        if self.warp_video_writer2D:
            self.warp_video_writer2D.release()
        if self.data_gradient_video_writer2D:
            self.data_gradient_video_writer2D.release()


