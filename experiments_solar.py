import matplotlib.pyplot as plt
import numpy as np
import pylops
from datetime import datetime
from glob import glob
from astropy.io import fits
import os
from itertools import chain
from tomograpy import project_3d, backproject_3d
from pylops.basicoperators import FunctionOperator
from pylops import LinearOperator, lsqr
from skimage.transform import resize


def z_rotation_matrix_3d(angle):
    return np.array([[[np.cos(angle), np.sin(angle), 0],
                      [-np.sin(angle), np.cos(angle), 0],
                      [0, 0, 1]]])


if __name__ == "__main__":
    print("Test started")
    start_time = datetime.now()

    angles = list(chain(range(70, 80), range(130, 140)))
    rad_angles = np.deg2rad(angles)

    path = "/Users/jhughes/Desktop/data/synthetic COMPLETE data/1 degree/"
    filenames = glob(path + "*.fits")
    loaded_imgs = {np.deg2rad(angle): fits.open(os.path.join(path + f"comp_wl_284_ang_{angle}.fits"))[0].data for angle in angles}

    radius = 300

    img_size = 100
    cube_size = 100

    densities = np.zeros((cube_size, cube_size, cube_size), dtype=np.float32)
    mask = np.ones((cube_size, cube_size, cube_size), dtype=bool)

    b = (0, 0, 0)
    b = (-cube_size / 2, -cube_size / 2, -cube_size / 2)

    delta = (1.0, 1.0, 1.0)
    path_distance = 500.0

    norms, xs, ys, zs, ds, imgs = [], [], [], [], [], []
    #angles = np.array([0, 30])
    for angle in rad_angles:
        t_angle = -angle + np.pi/2 #np.deg2rad(np.abs(90*np.cos(angle)))

        img_x = np.arange(img_size) - img_size / 2
        img_y = np.zeros((img_size, img_size))
        img_z = np.arange(img_size) - img_size / 2
        img_x, img_z = np.meshgrid(img_x, img_z)

        img_x, img_y, img_z = img_x.flatten(), img_y.flatten(), img_z.flatten()

        R = z_rotation_matrix_3d(t_angle)
        coords = (R @ np.stack([img_x, img_y, img_z]))[0]
        img_x, img_y, img_z = coords[0], coords[1], coords[2]
        img_x = radius * np.cos(angle) + img_x  # - img_size/2
        img_y = radius * np.sin(angle) + img_y  # - img_size/2

        xx = img_x.reshape((img_size, img_size)).astype(np.float32)
        yy = -img_y.reshape((img_size, img_size)).astype(np.float32)
        zz = img_z.reshape((img_size, img_size)).astype(np.float32)

        v1 = np.array([xx[0, 1] - xx[0, 0], yy[0, 1] - yy[0, 0], zz[0, 1] - zz[0, 0]])
        v2 = np.array([xx[1, 0] - xx[0, 0], yy[1, 0] - yy[0, 0], zz[1, 0] - zz[0, 0]])
        v1 = v1 / np.linalg.norm(v1)
        v2 = v2 / np.linalg.norm(v2)
        normal = np.cross(v1, v2)
        normal = normal / np.linalg.norm(normal)

        norm = normal  # R @ np.array([1E-7, 1E-7, -1.0])
        norm[norm == 0] = 1E-6
        norms.append(norm)
        print(np.rad2deg(angle), np.rad2deg(t_angle), norm)

        d = 500
        img = resize(loaded_imgs[angle], (img_size, img_size))  #project_3d(xx, yy, zz, densities, mask, b, delta, norm, d)
        xs.append(xx)
        ys.append(yy)
        zs.append(zz)
        ds.append(d)
        imgs.append(img.astype(np.float32))
    imgs = np.array(imgs)

    # show
    for angle, img in zip(rad_angles, imgs):
        fig, ax = plt.subplots()
        im = ax.imshow(img)
        fig.colorbar(im)
        fig.savefig(f"/Users/jhughes/Desktop/projection_solar/{int(np.rad2deg(angle)):03d}.png")
        plt.close()
    #
    # model = np.zeros(densities.shape, dtype=np.float32)
    # for i, img in enumerate(imgs.reshape(len(xs), xs[0].shape[0], xs[0].shape[1])):
    #     model = backproject_3d(xs[i], ys[i], zs[i], img,
    #                                 model,
    #                                 mask, b, delta, norms[i], ds[i], True)
    # total = np.sum(img)
    # print(total, img.shape, imgs.shape)
    # model = ((model - total) / (len(imgs) - 1)).astype(np.float32)

    # # # 3d plot
    # fig = plt.figure()
    # ax = fig.add_subplot(projection='3d')
    #
    # n = 100
    #
    # for x, y, z in zip(xs, ys, zs):
    #     ax.scatter(x, y, z)
    #
    # # for norm, sx, sy in zip(norms, shift_xs, shift_ys):
    # #     print("TEST", norm)
    # #     ax.plot([0+sx, 100*norm[0]+sx], [0, 100*norm[1]], [0+sy, sy+100*norm[2]])
    #
    # ax.set_xlabel('X Label')
    # ax.set_ylabel('Y Label')
    # ax.set_zlabel('Z Label')
    #
    # plt.show()


    class Tomo(LinearOperator):
        def __init__(self, xs, ys, zs, norms, ds, b, delta, model_shape, mask, dtype=None):
            self.xs = xs
            self.ys = ys
            self.zs = zs
            self.norms = norms
            self.ds = ds
            self.b = b
            self.delta = delta
            self.model_shape = model_shape
            self.mask = mask
            super().__init__(dtype=np.dtype(dtype),
                             dims=self.model_shape,
                             dimsd=(len(self.xs), self.xs[0].shape[0], self.xs[0].shape[1]))

        def _matvec(self, densities):
            return np.array([project_3d(x, y, z, densities.reshape(self.model_shape).astype(np.float32),
                                        self.mask, self.b, self.delta, norm, d)
                    for x, y, z, norm, d in zip(self.xs, self.ys, self.zs, self.norms, self.ds)]).flatten()

        def _rmatvec(self, imgs):
            densitiesi = np.zeros(self.model_shape, dtype=np.float32)
            for i, img in enumerate(imgs.reshape(len(self.xs), self.xs[0].shape[0], self.xs[0].shape[1])):
                # densitiesi += backproject_3d(self.xs[i], self.ys[i], self.zs[i], img,
                #                densitiesi,
                #                self.mask, self.b, self.delta, self.norms[i], self.ds[i], True)
                densitiesi += backproject_3d(self.xs[i], self.ys[i], self.zs[i], img,
                                             np.zeros_like(densitiesi).astype(np.float32),
                                             self.mask, self.b, self.delta, self.norms[i], self.ds[i], True)
            # return ((densitiesi - np.sum(img)) / (len(self.xs)-1)).astype(np.float32)
            return densitiesi.flatten().astype(np.float32) #/ densitiesi.shape[0] / len(self.xs)
            #return densitiesi.flatten().astype(np.float32)

    print(len(xs))
    op = Tomo(xs, ys, zs, norms, ds, b, delta, densities.shape, mask, dtype=np.float32)

    Dop = [
        pylops.FirstDerivative(
            (cube_size, cube_size, cube_size), axis=0, edge=False, kind="backward", dtype=np.float32
        ),
        pylops.FirstDerivative(
            (cube_size, cube_size, cube_size), axis=1, edge=False, kind="backward", dtype=np.float32
        ),
        pylops.FirstDerivative(
            (cube_size, cube_size, cube_size), axis=2, edge=False, kind="backward", dtype=np.float32
        )
    ]
    #
    # # TV
    # mu = 1.5
    # lamda = [0.1, 0.1, 0.1]
    # niter_out = 2
    # niter_in = 1
    #
    # model = pylops.optimization.sparsity.splitbregman(
    #     op,
    #     imgs.ravel(),
    #     Dop,
    #     niter_outer=niter_out,
    #     niter_inner=niter_in,
    #     mu=mu,
    #     epsRL1s=lamda,
    #     tol=1e-4,
    #     tau=1.0,
    #     show=True,
    #     **dict(iter_lim=5, damp=1e-4)
    # )[0]

    # model = op / imgs.flatten()
    # model = pylops.optimization.leastsquares.regularized_inversion(
    #     op, imgs.flatten(), Dop, **dict(iter_lim=10, show=True, atol=1E-8, btol=1E-8)
    # )[0]
    #
    # model = pylops.optimization.basic.lsqr(op, imgs.flatten(),  x0=np.random.rand(*densities.flatten().shape).astype(np.float32),
    #                                        niter=100, show=True, damp=0)[0]
    model = pylops.optimization.basic.lsqr(op, imgs.flatten(), niter=3, show=True)[0]#, x0 = np.random.randint(0, 30, densities.shape))[0] # x0=densities.flatten() + 50 * np.random.rand(*densities.flatten().shape).astype(np.float32) - 25)[0]
                                           #                                        niter=100, show=True, damp=0)[0]
    #model = pylops.optimization.basic.lsqr(op, imgs.flatten(), x0=densities.flatten(), niter=10, show=True)[0]
    # model = pylops.optimization.basic.lsqr(op, imgs.flatten(), niter=1, show=True)[0]

    # from pylops.optimization.cls_sparsity import FISTA
    #
    # fistasolver = FISTA(op)
    #
    # model = fistasolver.solve(imgs.flatten(), niter=2, show=True)[0]

    #model = pylops.optimization.basic.cgls(op, imgs.flatten(), niter=10, show=True)[0]
    model = model.reshape(densities.shape)

    np.save("/Users/jhughes/Desktop/projection_solar/cube.npy", model)
    limit = np.nanpercentile(model, 95)

    for i in range(cube_size):
        fig, ax = plt.subplots()
        im = ax.imshow(model[:, :, i], vmin=0, vmax=limit)
        fig.colorbar(im)
        fig.show()
        fig.savefig(f"/Users/jhughes/Desktop/projection_solar/test_{i:03d}.png")

    reconstructions = op @ model.flatten()
    reconstructions = reconstructions.reshape((len(xs), img_size, img_size))

    for i, angle in enumerate(rad_angles):
        fig, axs = plt.subplots(ncols=2)
        im = axs[0].imshow(imgs[i])
        fig.colorbar(im)
        im = axs[1].imshow(reconstructions[i])
        fig.colorbar(im)
        fig.show()
        fig.savefig(f"/Users/jhughes/Desktop/projection_solar/recon_{np.rad2deg(angle)}.png")



    end_time = datetime.now()
    print(end_time - start_time)
