import os, math, random, cv2, shutil
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(0)

def generate_lq(sample_factor):
    dataroot = "sss2depth"
    length = 964
    depth_slices = []
    sss_slices = []

    for i in range(length):
        sample_path = os.path.join(dataroot, str(i))
        d = np.load(os.path.join(sample_path, "d.npy"))
        s = np.load(os.path.join(sample_path, "s.npy"))

        d_lq = d

        for i in range(d.shape[0]):
            if i % sample_factor != 0:
                for j in range(d.shape[1]):
                    d_lq[i][j] = 0

        np.save(os.path.join(sample_path, "d_" + str(sample_factor) + ".npy"), d_lq)

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

def save_subset(path, idx_list):
    dataroot = "sss2depth"
    splitroot = "sss2depth_split"

    length = 964
    subset_length = idx_list.shape[0]

    for i in range(subset_length):
        sample_path = os.path.join(dataroot, str(idx_list[i]))
        os.system("cp -r " + sample_path + " " + path + "/" + str(i))

def split_dataset():
    dataroot = "sss2depth"
    length = 964

    splitroot = "sss2depth_split"
    # shutil.rmtree(splitroot)
    os.mkdir(splitroot)
    os.mkdir(splitroot + "/train")
    os.mkdir(splitroot + "/validation")
    os.mkdir(splitroot + "/test")

    index = np.arange(length)
    np.random.shuffle(index)

    # Let's divide it like 80% for training, 10% val and 10% test
    nbr_train = int(0.8*length)
    nbr_val = int(0.1*length)

    save_subset(splitroot + "/train", index[:nbr_train])
    save_subset(splitroot + "/validation", index[nbr_train:nbr_train+nbr_val])
    save_subset(splitroot + "/test", index[nbr_train+nbr_val:])

def remove():
    dataroot = "sss2depth"
    length = 964

    for i in range(length):
        sample_path = os.path.join(dataroot, str(i))
        os.remove(os.path.join(sample_path, "b.npy"))

 
if __name__ == "__main__":
    # generate different lq
    n = 4
    # generate_lq(n)
    show_images(0, 5, n)

    # split 8:1:1
    split_dataset()
