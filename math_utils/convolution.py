#  ================================================================
#  Created by Gregory Kramida on 9/18/18.
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
import numpy as np
from utils.sampling import get_focus_coordinates
from utils.printing import *

sobolev_kernel_1d = np.array([2.995900285895913839e-04,
                              4.410949535667896271e-03,
                              6.571318954229354858e-02,
                              9.956527948379516602e-01,
                              6.571318954229354858e-02,
                              4.410949535667896271e-03,
                              2.995900285895913839e-04])


def convolve_with_kernel_y(vector_field, kernel):
    y_convolved = np.zeros_like(vector_field)
    if len(vector_field.shape) == 3 and vector_field.shape[2] == 2:
        for x in range(vector_field.shape[1]):
            y_convolved[:, x, 0] = np.convolve(vector_field[:, x, 0], kernel, mode='same')
            y_convolved[:, x, 1] = np.convolve(vector_field[:, x, 1], kernel, mode='same')
        np.copyto(vector_field, y_convolved)
    elif len(vector_field.shape) == 4 and vector_field.shape[3] == 3:
        for z in range(vector_field.shape[2]):
            for x in range(vector_field.shape[0]):
                for i_val in range(3):
                    y_convolved[x, :, z, i_val] = np.convolve(vector_field[x, :, z, i_val], kernel, mode='same')
    else:
        raise ValueError("Can only process tensors with 3 dimensions (where last dimension is 2) or "
                         "tensors with 4 dimensions (where last dimension is 3), i.e. 2D & 3D vector fields")
    return y_convolved


def convolve_with_kernel_x(vector_field, kernel):
    x_convolved = np.zeros_like(vector_field)
    if len(vector_field.shape) == 3 and vector_field.shape[2] == 2:
        for y in range(vector_field.shape[0]):
            x_convolved[y, :, 0] = np.convolve(vector_field[y, :, 0], kernel, mode='same')
            x_convolved[y, :, 1] = np.convolve(vector_field[y, :, 1], kernel, mode='same')
    elif len(vector_field.shape) == 4 and vector_field.shape[3] == 3:
        for z in range(vector_field.shape[0]):
            for y in range(vector_field.shape[1]):
                for i_val in range(3):
                    x_convolved[z, y, :, i_val] = np.convolve(vector_field[z, y, :, i_val], kernel, mode='same')
    else:
        raise ValueError("Can only process tensors with 3 dimensions (where last dimension is 2) or "
                         "tensors with 4 dimensions (where last dimension is 3), i.e. 2D & 3D vector fields")
    np.copyto(vector_field, x_convolved)
    return x_convolved


def convolve_with_kernel_z(vector_field, kernel):
    if len(vector_field.shape) != 4 or vector_field.shape[3] != 3:
        raise ValueError("Can only process tensors with 4 dimensions (where last dimension is 3), i.e. 3D Vector field")


def convolve_with_kernel(vector_field, kernel=sobolev_kernel_1d, print_focus_coord_info=False):
    x_convolved = np.zeros_like(vector_field)
    y_convolved = np.zeros_like(vector_field)
    z_convolved = None
    if len(vector_field.shape) == 3 and vector_field.shape[2] == 2:
        focus_coordinates = get_focus_coordinates()

        for x in range(vector_field.shape[1]):
            y_convolved[:, x, 0] = np.convolve(vector_field[:, x, 0], kernel, mode='same')
            y_convolved[:, x, 1] = np.convolve(vector_field[:, x, 1], kernel, mode='same')

        for y in range(vector_field.shape[0]):
            x_convolved[y, :, 0] = np.convolve(y_convolved[y, :, 0], kernel, mode='same')
            x_convolved[y, :, 1] = np.convolve(y_convolved[y, :, 1], kernel, mode='same')

        if print_focus_coord_info:
            new_gradient_at_focus = vector_field[focus_coordinates[1], focus_coordinates[0]]
            print(
                " H1 grad: {:s}[{:f} {:f}{:s}]".format(BOLD_GREEN, -new_gradient_at_focus[0], -new_gradient_at_focus[1],
                                                       RESET), sep='', end='')
        np.copyto(vector_field, x_convolved)

    elif len(vector_field.shape) == 4 and vector_field.shape[3] == 3:
        z_convolved = np.zeros_like(vector_field)
        for z in range(vector_field.shape[0]):
            for y in range(vector_field.shape[1]):
                for i_val in range(3):
                    x_convolved[z, y, :, i_val] = np.convolve(vector_field[z, y, :, i_val], kernel, mode='same')
        for z in range(vector_field.shape[0]):
            for x in range(vector_field.shape[2]):
                for i_val in range(3):
                    y_convolved[z, :, x, i_val] = np.convolve(x_convolved[z, :, x, i_val], kernel, mode='same')
        for y in range(vector_field.shape[1]):
            for x in range(vector_field.shape[2]):
                for i_val in range(3):
                    z_convolved[:, y, x, i_val] = np.convolve(y_convolved[:, y, x, i_val], kernel, mode='same')
        np.copyto(vector_field, z_convolved)
    else:
        raise ValueError("Can only process tensors with 3 dimensions (where last dimension is 2) or "
                         "tensors with 4 dimensions (where last dimension is 3), i.e. 2D & 3D vector fields")

    return vector_field


def convolve_with_kernel_preserve_zeros(vector_field, kernel=sobolev_kernel_1d, print_focus_coord_info=False):
    x_convolved = np.zeros_like(vector_field)
    y_convolved = np.zeros_like(vector_field)
    focus_coordinates = get_focus_coordinates()
    zero_check = np.abs(vector_field) < 1e-6
    for x in range(vector_field.shape[1]):
        y_convolved[:, x, 0] = np.convolve(vector_field[:, x, 0], kernel, mode='same')
        y_convolved[:, x, 1] = np.convolve(vector_field[:, x, 1], kernel, mode='same')
    y_convolved[zero_check] = 0.0
    for y in range(vector_field.shape[0]):
        x_convolved[y, :, 0] = np.convolve(y_convolved[y, :, 0], kernel, mode='same')
        x_convolved[y, :, 1] = np.convolve(y_convolved[y, :, 1], kernel, mode='same')
    x_convolved[zero_check] = 0.0
    np.copyto(vector_field, x_convolved)
    if print_focus_coord_info:
        new_gradient_at_focus = vector_field[focus_coordinates[1], focus_coordinates[0]]
        print(" H1 grad: {:s}[{:f} {:f}{:s}]".format(BOLD_GREEN, -new_gradient_at_focus[0], -new_gradient_at_focus[1],
                                                     RESET), sep='', end='')
    return vector_field
