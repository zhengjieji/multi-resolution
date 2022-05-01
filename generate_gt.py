from auvlib.bathy_maps import map_draper

import os, math, random, cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image


def convert_waterfall_to_slices(wf_sss, wf_depth, image_height=32):
    image_width = int(wf_sss.shape[1]/2)
    wf_length = wf_sss.shape[0]

    sss_slices = []
    depth_slices = []

    for i in range(0, wf_length-image_height, int(image_height/2)):
        sss_slice_left = wf_sss[i:i+image_height, :image_width] + 10.
        depth_slice_left = wf_depth[i:i+image_height, :image_width]
        sss_slice_right = wf_sss[i:i+image_height, image_width:] + 10.
        depth_slice_right = wf_depth[i:i+image_height, image_width:]

        sss_slices.append(sss_slice_right)
        depth_slices.append(depth_slice_right)
        sss_slices.append(np.fliplr(sss_slice_left))
        depth_slices.append(np.fliplr(depth_slice_left))

    return sss_slices, depth_slices

def rescale_slices(slices):
    # slices = [np.array(Image.fromarray((255.*np.rot90(s)).astype(np.uint8)).resize((256, 256), Image.ANTIALIAS)) for s in slices]
    slices = [np.rot90(s).reshape((256, 256)) for s in slices]

    return slices

def normalize_intensities(sss_slices):
    new_slices = []
    for s in sss_slices:
        m = np.mean(s[s>.1])
        ss = s.copy()
        if m != 0 and not np.isnan(m):
            ss *= .3/m
        new_slices.append(ss)

    return new_slices

def generate_gt():
    folders = [2,3,4,5,6,12,13,14,15]

    all_map_images = []
    for n in folders:
        map_images = map_draper.sss_map_image.read_data(str(n) + ".cereal")
        all_map_images.extend(map_images)

    image_height = 256
    image_width =all_map_images[0].sss_waterfall_image.shape[1]/2

    # Let's start by cutting the waterfall images into slices
    sss_slices = []
    depth_slices = []
    for map_image in all_map_images:
        s, d = convert_waterfall_to_slices(map_image.sss_waterfall_image, map_image.sss_waterfall_depth, image_height)
        sss_slices.extend(s)
        depth_slices.extend(d)

    # normalize the depth slices to an interval
    max_depth = np.max(depth_slices)
    min_depth = np.min(depth_slices)
    depth_slices = [1./(max_depth - min_depth)*(d - min_depth*(d != 0).astype(np.float)) for d in depth_slices]

    # filter slices which are too dark (too deep)
    min_depth_mean = -0 # 0.3

    sss_slices, depth_slices = map(list,zip(*[(s, d) for s, d in zip(sss_slices, depth_slices) if np.mean(d) > min_depth_mean]))

    # make the sidescan images look good by normalizing them
    sss_slices = normalize_intensities(sss_slices)

    # convert to 256x256 8 bit grayscale images
    depth_slices = rescale_slices(depth_slices)

    # The directory where we'll save the dataset
    dataroot = "sss2depth"

    # This will fail if it already exists, that's good
    os.makedirs(dataroot)
    length = len(sss_slices)

    for i in range(length):

        sample_path = os.path.join(dataroot, str(i))
        os.makedirs(sample_path)

        d = depth_slices[i]
        s = sss_slices[i]
        s = cv2.rotate(s, cv2.ROTATE_90_COUNTERCLOCKWISE) # rotate 90 for correspondence

        np.save(os.path.join(sample_path, "d.npy"), d)
        np.save(os.path.join(sample_path, "s.npy"), s)

def data_augmentation():
    dataroot = "sss2depth"
    length = 964
    depth_slices = []
    sss_slices = []

    for i in range(length):
        sample_path = os.path.join(dataroot, str(i))
        d = np.load(os.path.join(sample_path, "d.npy"))
        s = np.load(os.path.join(sample_path, "s.npy"))

        if i % 5 == 0:
            d_flip = cv2.flip(d, 1)
            s_flip = cv2.flip(s, 1)
            np.save(os.path.join(sample_path, "d.npy"), d_flip)
            np.save(os.path.join(sample_path, "s.npy"), s_flip)

    for i in range(length):
        sample_path = os.path.join(dataroot, str(i))
        d = np.load(os.path.join(sample_path, "d.npy"))
        s = np.load(os.path.join(sample_path, "s.npy"))

        if i % 4 == 0:
            row, col = d.shape
            mean = 0
            var = 0.00001
            sigma = var ** 0.5
            gauss = np.random.normal(mean, sigma, (row,col))
            gauss = gauss.reshape(row,col)
            noisy_d = d + gauss
            np.save(os.path.join(sample_path, "d.npy"), noisy_d)
    
def show_images(index, n):
    dataroot = "sss2depth"
    length = 964
    depth_slices = []
    sss_slices = []
    for i in range(length):
        sample_path = os.path.join(dataroot, str(i))
        d = np.load(os.path.join(sample_path, "d.npy"))
        s = np.load(os.path.join(sample_path, "s.npy"))
        depth_slices.append(d)
        sss_slices.append(s)

    depth_slices = np.array(depth_slices)
    sss_slices = np.array(sss_slices)

    for s, d in zip(sss_slices[index:index+n], depth_slices[index:index+n]):
        fig, ax = plt.subplots(1, 2)
        ax[0].imshow(s)
        ax[0].set_title("Sidescan")
        ax[1].imshow(d)
        ax[1].set_title("Depth")
        plt.show()


if __name__ == "__main__":
    generate_gt()
    data_augmentation()
    show_images(0, 5)