import matplotlib.pyplot as plt
import numpy as np
import pylops
import scipy.fftpack as fp
from tomograpy import project_3d, backproject_3d


def z_rotation_matrix_3d(angle):
    return np.array([[[np.cos(angle), np.sin(angle), 0],
                      [-np.sin(angle), np.cos(angle), 0],
                      [0, 0, 1]]])


angles = np.linspace(0, 2*np.pi, 50)

radius = 500

img_size = 100
cube_size = 100
# x, y = np.meshgrid(np.arange(img_size, dtype=np.float32), np.arange(img_size, dtype=np.float32))
# z = np.zeros((img_size, img_size), dtype=np.float32)
densities = np.zeros((cube_size, cube_size, cube_size), dtype=np.float32)
# densities[1:30, 1:30, 1:30] = 25
# densities[80:90, 80:90, 80:90] = 5
densities[10:-10, 10:-10, 10:-10] = 100
densities[30:-30, 30:-30, 30:-30] = 0


# densities[2, 2, 2] = 5
# densities[8, 8, 2] = 8
# densities[10:30, 10:30, 0:30] = 3
mask = np.ones((cube_size, cube_size, cube_size), dtype=bool)

b = (0, 0, 0)
b = (-cube_size / 2, -cube_size / 2, -cube_size / 2)

delta = (1.0, 1.0, 1.0)
# unit_normal = (1E-7, 1E-7, 1.0)
path_distance = 500.0

total = np.zeros_like(densities)

for angle in angles:
    t_angle = -angle + np.pi / 2  # np.deg2rad(np.abs(90*np.cos(angle)))

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
    print(np.rad2deg(angle), np.rad2deg(t_angle), norm)

    proj = project_3d(xx, yy, zz, densities, mask, b, delta, norm, path_distance)

    fig, ax = plt.subplots()
    im = ax.imshow(proj, origin='lower')
    fig.colorbar(im)
    fig.savefig(f"/Users/jhughes/Desktop/projection_bpj/proj_{angle:0.3f}.png")
    plt.close()

    # FILTERING
    # (w, h) = proj.shape
    # half_w, half_h = int(w / 2), int(h / 2)
    #
    # F1 = fp.fft2(proj.astype(float))
    # F2 = fp.fftshift(F1)
    #
    # # high pass filter
    # n = 10
    # F2[half_w - n:half_w + n + 1, half_h - n:half_h + n + 1] = 0
    # proj = fp.ifft2(fp.ifftshift(F2)).real

    result = backproject_3d(xx, yy, zz, proj.astype(np.float32),
                       np.zeros_like(densities, dtype=np.float32),
                       mask, b, delta, norm, path_distance, True)
    print(result.shape)

    scaling = densities.shape[0]
    result = result / scaling

    total += result
# ((result - np.sum(proj)) / (len(self.xs)-1)).astype(np.float32)
#result = result - np.sum(proj)

# # Scale result to make sure that fbp(A, A(x)) == x holds at least
# # to some approximation. In limited experiments, this is true for
# # this version of FBP up to 1%.
# # *Note*: For some reason, we do not have to scale with respect to
# # the pixel dimension that is orthogonal to the rotation axis (`u`
# # or horizontal pixel dimension). Hence, we only scale with the
# # other pixel dimension (`v` or vertical pixel dimension).
# vg, pg = A.astra_compat_vg, A.astra_compat_pg
#
# pixel_height = (pg.det_size[0] / pg.det_shape[0])
# voxel_volume = np.prod(np.array(vg.size / np.array(vg.shape)))
# scaling = (np.pi / pg.num_angles) * pixel_height / voxel_volume
#
# rec *= scaling
# pixel_height = proj.size / proj.shape[0]
# voxel_volume = np.prod(densities.size / np.array(densities.shape))
# scaling = (np.pi / 1) * pixel_height / voxel_volume

total /= len(angles)

for i in range(cube_size):
    fig, axs = plt.subplots(ncols=2)
    im = axs[0].imshow(densities[i, :, :], origin='lower', vmin=0, vmax=150)
    fig.colorbar(im)

    im = axs[1].imshow(total[i, :, :], origin='lower')#, vmin=0, vmax=150)
    fig.colorbar(im)
    fig.savefig(f"/Users/jhughes/Desktop/projection_bpj/{i:03d}.png")
    plt.close()
