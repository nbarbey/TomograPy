import matplotlib.pyplot as plt
import numpy as np
import pylops
from datetime import datetime

from tomograpy import project_3d, backproject_3d
from pylops.basicoperators import FunctionOperator
from pylops import LinearOperator, lsqr


def z_rotation_matrix_3d(angle):
    return np.array([[[np.cos(angle), np.sin(angle), 0],
                      [-np.sin(angle), np.cos(angle), 0],
                      [0, 0, 1]]])


if __name__ == "__main__":
    print("Test started")
    start_time = datetime.now()

    angles = np.linspace(0, np.pi, 14)

    radius = 300

    img_size = 20
    cube_size = 20
    densities = np.zeros((cube_size, cube_size, cube_size), dtype=np.float32)
    densities[3:-3, 3:-3, 3:-3] = 100
    densities[7:-7, 7:-7, 7:-7] = 0
    # densities = np.random.randint(0, 100, (cube_size, cube_size, cube_size)).astype(np.float32)
    # densities[5, 5, :] = 0

    mask = np.ones((cube_size, cube_size, cube_size), dtype=bool)

    b = (0, 0, 0)
    b = (-cube_size / 2, -cube_size / 2, -cube_size / 2)

    delta = (1.0, 1.0, 1.0)
    path_distance = 500.0

    norms, xs, ys, zs, ds, imgs = [], [], [], [], [], []
    for angle in angles:
        t_angle = -angle + np.pi/2

        img_x = np.arange(img_size) - img_size / 2
        img_y = np.zeros((img_size, img_size))
        img_z = np.arange(img_size) - img_size / 2
        img_x, img_z = np.meshgrid(img_x, img_z)

        img_x, img_y, img_z = img_x.flatten(), img_y.flatten(), img_z.flatten()

        R = z_rotation_matrix_3d(t_angle)
        coords = (R @ np.stack([img_x, img_y, img_z]))[0]
        img_x, img_y, img_z = coords[0], coords[1], coords[2]
        img_x = radius * np.cos(angle) + img_x
        img_y = radius * np.sin(angle) + img_y

        xx = img_x.reshape((img_size, img_size)).astype(np.float32)
        yy = -img_y.reshape((img_size, img_size)).astype(np.float32)
        zz = img_z.reshape((img_size, img_size)).astype(np.float32)

        v1 = np.array([xx[0, 1] - xx[0, 0], yy[0, 1] - yy[0, 0], zz[0, 1] - zz[0, 0]])
        v2 = np.array([xx[1, 0] - xx[0, 0], yy[1, 0] - yy[0, 0], zz[1, 0] - zz[0, 0]])
        v1 = v1 / np.linalg.norm(v1)
        v2 = v2 / np.linalg.norm(v2)
        normal = np.cross(v1, v2)
        normal = normal / np.linalg.norm(normal)

        norm = normal
        norm[norm == 0] = 1E-6
        norms.append(norm)
        print(np.rad2deg(angle), np.rad2deg(t_angle), norm)

        d = 500
        img = project_3d(xx, yy, zz, densities, mask, b, delta, norm, d)
        xs.append(xx)
        ys.append(yy)
        zs.append(zz)
        ds.append(d)
        imgs.append(img.astype(np.float32))
    imgs = np.array(imgs)

    # show
    for angle, img in zip(angles, imgs):
        fig, ax = plt.subplots()
        im = ax.imshow(img)
        fig.colorbar(im)
        fig.savefig(f"/Users/jhughes/Desktop/projection_dots/{int(np.rad2deg(angle)):03d}.png")
        plt.close()


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
                             dimsd=(1, self.xs[0].shape[0], self.xs[0].shape[1]))
        #                     dimsd=(len(self.xs), self.xs[0].shape[0], self.xs[0].shape[1]))
        #
        # def _matvec(self, densities):
        #     return np.array([project_3d(x, y, z, densities.reshape(self.model_shape).astype(np.float32),
        #                                 self.mask, self.b, self.delta, norm, d)
        #             for x, y, z, norm, d in zip(self.xs, self.ys, self.zs, self.norms, self.ds)]).flatten()
        #
        # def _rmatvec(self, imgs):
        #     densitiesi = np.zeros(self.model_shape, dtype=np.float32)
        #     for i, img in enumerate(imgs.reshape(len(self.xs), self.xs[0].shape[0], self.xs[0].shape[1])):
        #         # densitiesi += backproject_3d(self.xs[i], self.ys[i], self.zs[i], img,
        #         #                densitiesi,
        #         #                self.mask, self.b, self.delta, self.norms[i], self.ds[i], True)
        #         densitiesi = backproject_3d(self.xs[i], self.ys[i], self.zs[i], img,
        #                                      densitiesi,
        #                                      self.mask, self.b, self.delta, self.norms[i], self.ds[i], True)
        #     # return ((densitiesi - np.sum(img)) / (len(self.xs)-1)).astype(np.float32)
        #     return densitiesi / (cube_size - 1) / (len(self.xs) / 2 + 1)#/ len(self.xs) #/ densitiesi.shape[0] / len(self.xs)
        #     #return densitiesi.flatten().astype(np.float32)

        def _matvec(self, densities):
            # return np.array([project_3d(x, y, z, densities.reshape(self.model_shape).astype(np.float32),
            #                             self.mask, self.b, self.delta, norm, d)
            #         for x, y, z, norm, d in zip(self.xs, self.ys, self.zs, self.norms, self.ds)]).flatten()
            i = 0
            return project_3d(self.xs[i], self.ys[i], self.zs[i], densities.reshape(self.model_shape).astype(np.float32), self.mask, self.b, self.delta, self.norms[i], self.ds[i]).flatten()

        def _rmatvec(self, imgs):
            densitiesi = np.zeros(self.model_shape, dtype=np.float32)
            for i, img in enumerate(imgs.reshape(1, self.xs[0].shape[0], self.xs[0].shape[1])):
                # densitiesi += backproject_3d(self.xs[i], self.ys[i], self.zs[i], img,
                #                densitiesi,
                #                self.mask, self.b, self.delta, self.norms[i], self.ds[i], True)
                # if i == 0:
                densitiesi += backproject_3d(self.xs[i], self.ys[i], self.zs[i], img,
                                             np.zeros_like(densitiesi).astype(np.float32),
                                             self.mask, self.b, self.delta, self.norms[i], self.ds[i], True)
            # return ((densitiesi - np.sum(img)) / (len(self.xs)-1)).astype(np.float32)
            #return (densitiesi / (cube_size - 1)).flatten() #/ densitiesi.shape[0] / len(self.xs)
            return densitiesi.flatten().astype(np.float32)

    print(len(xs))
    op = Tomo(xs, ys, zs, norms, ds, b, delta, densities.shape, mask, dtype=np.float32)

    proj = op @ densities.flatten()
    densities_bpj = op.H @ proj.flatten()
    proj2 = op @ densities_bpj.flatten()
    densities_bpj2 = op.H @ proj2.flatten()
    print(np.where(proj == -999))

    #print(np.conj(proj).T @ (op @ densities))

    print("maxs", np.max(proj), np.max(proj2))
    print("pcts", np.nanpercentile(densities, 95), np.nanpercentile(densities_bpj, 95),  np.nanpercentile(densities_bpj2, 95))
    print("close", np.allclose(densities_bpj, densities_bpj2))

    fig, axs = plt.subplots(ncols=2)
    im = axs[0].imshow(proj.reshape((img_size, img_size)), vmin=0, vmax=1300)
    fig.colorbar(im)
    im = axs[1].imshow(proj2.reshape((img_size, img_size))/ (cube_size-1))#, vmin=0, vmax=1300)
    fig.colorbar(im)
    fig.savefig("/Users/jhughes/Desktop/projection_dots/comparison.png")

    fig, axs = plt.subplots(ncols=3)
    im = axs[0].imshow(densities.reshape((cube_size, cube_size, cube_size))[0], vmin=0, vmax=100)
    fig.colorbar(im)
    im = axs[1].imshow(densities_bpj.reshape((cube_size, cube_size, cube_size))[0])# , vmin=0, vmax=100)
    fig.colorbar(im)
    im = axs[2].imshow(densities_bpj2.reshape((cube_size, cube_size, cube_size))[0])# , vmin=0, vmax=100)
    fig.colorbar(im)
    fig.savefig("/Users/jhughes/Desktop/projection_dots/comparison_dense.png")


    from pylops.utils import dottest
    _ = dottest(op, 400, 8000, atol=0.1, complexflag=0, verb=True)

    model = pylops.optimization.basic.lsqr(op, imgs[0].flatten(), niter=10, show=True)[0]
    #
    # model = model.reshape(densities.shape)
    #
    # limit = np.nanpercentile(model, 95)
    #
    # for i in range(cube_size):
    #     fig, axs = plt.subplots(ncols=2)
    #     im = axs[0].imshow(densities[:, :, i], vmin=0, vmax=150)#, vmin=0, vmax=5)
    #     fig.colorbar(im)
    #     im = axs[1].imshow(model[:, :, i], vmin=0, vmax=limit)
    #     fig.colorbar(im)
    #     fig.show()
    #     fig.savefig(f"/Users/jhughes/Desktop/projection_dots/test_{i:03d}.png")
