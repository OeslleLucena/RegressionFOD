from __future__ import absolute_import, print_function
import numpy as np
import pyshtools
from niftynet.layer.base_layer import RandomisedLayer


class FODRandomRotationLayer(RandomisedLayer):
    """
    generate randomised FOD rotations for data augmentation
    """
    def __init__(self, name='random_rotation', lmax=4):
        super(FODRandomRotationLayer, self).__init__(name=name)
        self.min_angle = 0.0
        self.max_angle = 0.0
        self.alpha = 0.0 # rotation around the initial z-axis
        self.beta = 0.0  # rotation around the new y-axis
        self.gamma = 0.0 # rotation around the new z-axis
        self.dj_rotation_matrix = [[0.0, 0.0,0.0]]
        self.lmax = lmax
        self.num_coefficients = np.int((self.lmax + 1) * (self.lmax + 2) / 2)

        #TODO: make this general to any lmax order
        self.coefficients_matrix_indices = np.asarray([[0, 0, 0],
                                                      [1, 2, 2],
                                                      [1, 2, 1],
                                                      [0, 2, 0],
                                                      [0, 2, 1],
                                                      [0, 2, 2],
                                                      [1, 4, 4],
                                                      [1, 4, 3],
                                                      [1, 4, 2],
                                                      [1, 4, 1],
                                                      [0, 4, 0],
                                                      [0, 4, 1],
                                                      [0, 4, 2],
                                                      [0, 4, 3],
                                                      [0, 4, 4]])

        self.coefficients_matrix_indices = (self.coefficients_matrix_indices[:, 0],
                                            self.coefficients_matrix_indices[:, 1],
                                            self.coefficients_matrix_indices[:, 2])

        self.num_coefficients = np.int((self.lmax + 1) * (self.lmax + 2) / 2)
        self.coefficients_indices = np.arange(self.num_coefficients)
        self.coefficients = np.zeros((self.num_coefficients,))
        self.coefficients_matrix = np.zeros((2, self.lmax + 1, self.lmax + 1))
        self.dj_rotation_matrix = pyshtools.rotate.djpi2(self.lmax)


    def init_uniform_angle(self, rotation_angle=(-10.0, 10.0)):
        assert rotation_angle[0] < rotation_angle[1]
        self.min_angle = float(rotation_angle[0])
        self.max_angle = float(rotation_angle[1])


    def randomise(self, spatial_rank=3):
        if spatial_rank == 3:
            self._randomise_sh_transformation_matrix()
        else:
            pass

    def _randomise_sh_transformation_matrix(self):
        alpha = np.random.uniform(self.min_angle, self.max_angle)
        beta = np.random.uniform(self.min_angle, self.max_angle)
        gamma = np.random.uniform(self.min_angle, self.max_angle)
        self.angles = np.radians([alpha, beta, gamma])

    def _generate_sh_matrix(self, coefficients):
        self.coefficients_matrix[self.coefficients_matrix_indices] = \
            coefficients[self.coefficients_indices]
        return self.coefficients_matrix

    def _sh_matrix_to_coeffs(self, coefficients_matrix):
        self.coefficients[self.coefficients_indices] = \
            coefficients_matrix[self.coefficients_matrix_indices]
        return self.coefficients

    def _apply_rotation_vectorized(self, x, y, z, image):
        coefficients_matrix = self._generate_sh_matrix(image[..., x, y, z])
        rotated_coefficients = pyshtools.rotate.SHRotateRealCoef(coefficients_matrix,
                                                                 self.angles,
                                                                 self.dj_rotation_matrix)
        image[..., x, y, z] =  self._sh_matrix_to_coeffs(rotated_coefficients)

    def _apply_rotation(self, image, idx=None):
        image = np.transpose(image, (3, 0, 1, 2))
        if idx is None:
            idx = np.nonzero(image[0, ...])

        x = idx[0]
        y = idx[1]
        z = idx[2]

        apply_rotation = np.vectorize(self._apply_rotation_vectorized,
                                      otypes=[float],
                                      excluded=[3])
        apply_rotation(x, y, z, image)

        image = np.transpose(image, (1, 2, 3, 0))

        return image

    def layer_op(self, inputs, interp_orders, *args, **kwargs):
        if inputs is None:
            return inputs

        if isinstance(inputs, dict) and isinstance(interp_orders, dict):
            #
            mask = inputs['sampler']
            idx = np.nonzero(mask[..., 0, :])
            for field in ['image', 'label']:
                inputs[field][..., 0, :] = \
                    self._apply_rotation(inputs[field][..., 0, :], idx)
        else:
            raise NotImplementedError("unknown input format")
        return inputs


