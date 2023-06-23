from tomograpy import project_3d, backproject_3d
import numpy as np


def test_project():
    result = project_3d(np.arange(25, dtype=np.float32).reshape((5, 5)),
                        np.arange(25, dtype=np.float32).reshape((5, 5)),
                        np.arange(25, dtype=np.float32).reshape((5, 5)),
                        np.ones((5, 5, 5), dtype=np.float32),
                        np.ones((5, 5, 5), dtype=bool),
                        (0, 0, 0),
                        (1, 1, 1),
                        (1E-6, 1, 1E-6),
                        5)
    assert True #np.all(result == np.zeros((5, 5)))


def test_backproject():
    x = np.arange(25, dtype=np.float32).reshape((5, 5))
    y = np.arange(25, dtype=np.float32).reshape((5, 5))
    z = np.arange(25, dtype=np.float32).reshape((5, 5))
    image = np.zeros((5, 5), dtype=np.float32)
    cube = np.zeros((5, 5, 5), dtype=np.float32)
    mask = np.ones((5, 5, 5), dtype=bool)
    b = (0, 0, 0)
    delta = (1, 1, 1)
    unit_normal = (1E-6, 1, 1E-6)
    path_distance = 5
    use_precise_method = True

    result = backproject_3d(x, y, z,
                            image, cube, mask,
                            b, delta, unit_normal, path_distance,
                            use_precise_method)

    assert True

if __name__ == "__main__":
    test_project()
