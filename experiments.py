import matplotlib.pyplot as plt
import numpy as np
import pylops

from tomograpy import project_3d, backproject_3d

if __name__ == "__main__":
    print("Test started")

    img_size = 50
    cube_size = 50
    x, y = np.meshgrid(np.arange(img_size, dtype=np.float32), np.arange(img_size, dtype=np.float32))
    z = np.zeros((img_size, img_size), dtype=np.float32) + 3
    densities = np.ones((cube_size, cube_size, cube_size), dtype=np.float32)
    # densities[1, 4, 1] = 100
    densities[2, 2, 2] = 5
    densities[8, 8, 2] = 8
    densities[10:30, 10:30, 0:30] = 3
    mask = np.ones((cube_size, cube_size, cube_size), dtype=bool)

    #b = (0.5, 0.5, 0.5)
    b = (0, 0, 0)
    #b = (1E-8, 1E-8, 1E-8)
    #b = (1, 1, 1)
    #b = (10, 10, 10)
    delta = (1, 1, 1)
    #unit_normal = (-1.0, -1E-8, -1E-8)
    #unit_normal = (1.0, 1E-8, 1E-8)
    #unit_normal = (-1, 0, 0)
    #unit_normal = (-1/np.sqrt(3), -1/np.sqrt(3), -1/np.sqrt(3))


    unit_normal = (1E-7, 1E-7, -1.0)
    path_distance = 500

    result = project_3d(x, y, z, densities, mask, b, delta, unit_normal, path_distance)
    # print(result)

    import pylops

    def forward(densities):
        return project_3d(x, y, z, densities.reshape((cube_size, cube_size, cube_size)).astype(np.float32),
                          mask, b, delta, unit_normal, path_distance).reshape((img_size**2))

    def backward(image):
        return backproject_3d(x, y, z, image.reshape((img_size, img_size)).astype(np.float32),
                   np.zeros_like(densities, dtype=np.float32),
                   mask, b, delta, unit_normal, path_distance, True).reshape((cube_size**3))

    def test_forward(xarr):
        print(xarr)
        return forward(xarr)

    from pylops.basicoperators import FunctionOperator

    op = pylops.FunctionOperator(forward, backward, img_size**2, cube_size**3)
    out = op @ densities.reshape((cube_size**3))

    fig, ax = plt.subplots()
    ax.imshow(out.reshape((img_size, img_size)))
    fig.show()
    # print(out.shape)
    # image = 2*np.ones((5, 5), dtype=np.float32)
    # image[2, 2] = 10

    image = out.reshape((img_size, img_size))


    import xarray as xr

    da = xr.DataArray(
        data=image.reshape((img_size*img_size)),
        dims = ["x"],
        coords = dict(
            lon=(["x"], x.reshape((img_size*img_size)))
        ),
        # dims=["x", "y"],
        # coords=dict(
        #     lon=(["x", "y"], x.reshape((img_size*img_size))),
        #     lat=(["x", "y"], y.reshape((img_size*img_size))),
        # ),
        attrs=dict(
            description="Ambient temperature.",
            units="degC",
        ),
    )


    xinv = op / image.reshape((img_size*img_size))

    # xinv = pylops.optimization.leastsquares.regularized_inversion(
    #     op, image.reshape((img_size*img_size)), [], **dict(damp=0, iter_lim=10, show=True, atol=1E-8, btol=1E-8)
    # )[0]
    # print(xinv.reshape((10, 10, 10)))

    fig, axs = plt.subplots(ncols=2, nrows=3, figsize=(10, 14))
    im = axs[0, 0].imshow(densities[:, :, 2], vmin=0, vmax=3)
    axs[0, 0].set_title("Input density slice")
    fig.colorbar(im)

    im = axs[1, 0].imshow(xinv.reshape((cube_size, cube_size, cube_size))[:, :, 2], vmin=0, vmax=3)
    axs[1, 0].set_title("Reconstructed density slice")
    fig.colorbar(im)

    im = axs[2, 0].imshow(densities[:, :, 2] - xinv.reshape((cube_size, cube_size, cube_size))[:, :, 2], vmin=-3, vmax=3, cmap='seismic')
    axs[2, 0].set_title("Input - reconstruction")
    fig.colorbar(im)

    im = axs[1, 1].imshow(image, vmin=0, vmax=100)
    axs[1, 1].set_title("Image used in reconstruction")
    fig.colorbar(im)

    axs[0, 1].set_axis_off()
    axs[2, 1].set_axis_off()
    plt.show()
    # BACKPROJECT
    # image = 2*np.ones((15, 15), dtype=np.float32)
    # result = backproject_3d(x, y, z, image,
    #                         np.zeros_like(densities, dtype=np.float32),
    #                         mask, b, delta, unit_normal, path_distance, True)
    # print("cube", result)
    #
    # import matplotlib as mpl
    # import matplotlib.pyplot as plt
    #
    # mpl.use('macosx')
    #
    # fig = plt.figure()
    # ax = fig.add_subplot(projection='3d')
    # x, y, z = np.where(result > 1)
    # ax.scatter(x, y, z)
    #
    # plt.show(block=True)

    print("Test ended")