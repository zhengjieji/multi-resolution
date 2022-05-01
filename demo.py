import os, math, random, cv2, shutil
import numpy as np
import matplotlib.pyplot as plt


def show_images(index, n, sample_factor):
    dataroot = "sss2depth"
    length = 964
    depth_slices = []
    sss_slices = []
    depth_lq_slices = []

    for i in range(length):
        sample_path = os.path.join(dataroot, str(i))
        d = np.load(os.path.join(sample_path, "d.npy"))
        s = np.load(os.path.join(sample_path, "s.npy"))
        d_lq = np.load(os.path.join(sample_path, "d_" + str(sample_factor) + ".npy"))
        depth_slices.append(d)
        sss_slices.append(s)
        depth_lq_slices.append(d_lq)

    depth_slices = np.array(depth_slices)
    sss_slices = np.array(sss_slices)
    depth_lq_slices = np.array(depth_lq_slices)

    for s, d, d_lq in zip(sss_slices[index:index+n], depth_slices[index:index+n], depth_lq_slices[index:index+n]):
        fig, ax = plt.subplots(1, 3)
        ax[0].imshow(s)
        ax[0].set_title("Sidescan")
        ax[1].imshow(d)
        ax[1].set_title("Depth")
        ax[2].imshow(d_lq)
        ax[2].set_title("Depth Low Quality")
        plt.show()

 
if __name__ == "__main__":
    # generate different lq
    n = 64

    # generate_lq(n)
    show_images(10, 1, n)

