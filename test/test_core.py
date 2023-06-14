from tomograpy import axpy, project_3d
import numpy as np


def test_basic():
    result = axpy(5., np.array([1., 2., 3.]), np.array([0., 0., 0.]))
    assert np.all(result == np.array([5, 10, 15]))


def test_project():
    result = project_3d(np.arange(25, dtype=np.float32).reshape((5,5)),
                        np.arange(25, dtype=np.float32).reshape((5,5)),
                        np.arange(25, dtype=np.float32).reshape((5,5)),
                        np.ones((5, 5, 5), dtype=np.float32),
                        np.zeros((5, 5, 5), dtype=bool),
                        (0, 0, 0),
                        (1, 1, 1),
                        (1E-6, 1E-6, 1E-6),
                        5)
    assert np.all(result == np.zeros((5, 5, 5)))


if __name__ == "__main__":
    test_basic()
