import math
import numpy as np
from scipy.spatial.transform import Rotation

def random_axis():
    a = 0
    b = 0
    while True:
        a = 2.0 * np.random.random() - 1.0
        b = 2.0 * np.random.random() - 1.0
        if a**2 + b**2 < 1:
            break

    axis = np.array([
        2.0 * a * np.sqrt(1.0 - a**2 - b**2),
        2.0 * b * np.sqrt(1.0 - a**2 - b**2),
        1.0 - 2.0 * (a**2 + b**2)
    ])

    return axis

def random_uniform_rotation(angle_max):
    while True:
        angle = np.cbrt(np.random.random() * angle_max**3)
        if np.random.random() < (np.sin(angle) / angle)**2:
            break

    axis = random_axis()

    return axis * angle

def random_direction(ndim=3):
    vec = np.random.randn(ndim)
    vec /= np.linalg.norm(vec)
    return vec

def sample_noise_pose(angle=math.pi/2., dist=0.1, degrees=False):
    rot_mat = Rotation.from_rotvec(angle*random_direction(), degrees=degrees).as_matrix()
    translation = dist*random_direction()

    return rot_mat, translation
