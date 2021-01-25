from skimage.morphology import ball, disk, dilation, binary_erosion, remove_small_objects, erosion, closing, \
    reconstruction, binary_closing
from skimage.measure import label, regionprops
from skimage.filters import roberts
from skimage.segmentation import clear_border
from scipy import ndimage as ndi
import cv2
import matplotlib.pyplot as plt
import numpy as np
import os
from PIL import Image
import shutil
from tqdm import tqdm
from numba import jit
import pandas as pd


def get_segmented_lungs(img_dir, plot=False, min_lung_area=3000, padding_num=0):
    """
    :param img_dir: Input image path
    :param plot: Whether to show pictures or not
    :param min_lung_area: The minimum possible area of a lung (as a threshold)
    :param padding_num: Interception needs padding size
    :return: True or False (img is useful or not)
    """

    src_img = cv2.imread(img_dir)
    img = Image.open(img_dir)
    img = np.array(img)[:, :, 0]
    img = img.astype(np.int16)
    img = img * 3
    img -= 1000
    im = img.copy()
    '''
    This funtion segments the lungs from the given 2D slice.
    '''
    if plot == True:
        f, plots = plt.subplots(8, 1, figsize=(5, 40))
    '''
    Step 1: Convert into a binary image. 
    '''
    binary = im < -600
    if plot == True:
        plots[0].axis('off')
        plots[0].imshow(binary, cmap=plt.cm.bone)
    '''
    Step 2: Remove the blobs connected to the border of the image.
    '''
    cleared = clear_border(binary)
    if plot == True:
        plots[1].axis('off')
        plots[1].imshow(cleared, cmap=plt.cm.bone)
    '''
    Step 3: Label the image.
    '''
    label_image = label(cleared)
    if plot == True:
        plots[2].axis('off')
        plots[2].imshow(label_image, cmap=plt.cm.bone)
    '''
    Step 4: Keep the labels with 2 largest areas.
    '''
    areas = [r.area for r in regionprops(label_image)]
    areas.sort()
    if len(areas) > 2:
        for region in regionprops(label_image):
            if region.area < areas[-2] or region.area < min_lung_area:
                for coordinates in region.coords:
                    label_image[coordinates[0], coordinates[1]] = 0
    binary = label_image > 0
    if plot == True:
        plots[3].axis('off')
        plots[3].imshow(binary, cmap=plt.cm.bone)
    '''
    Step 5: Erosion operation with a disk of radius 2. This operation is 
    seperate the lung nodules attached to the blood vessels.
    '''
    selem = disk(2)
    binary = binary_erosion(binary, selem)
    if plot == True:
        plots[4].axis('off')
        plots[4].imshow(binary, cmap=plt.cm.bone)
    '''
    Step 6: Closure operation with a disk of radius 10. This operation is 
    to keep nodules attached to the lung wall.
    '''
    selem = disk(10)
    binary = binary_closing(binary, selem)
    if plot == True:
        plots[5].axis('off')
        plots[5].imshow(binary, cmap=plt.cm.bone)
    '''
    Step 7: Fill in the small holes inside the binary mask of lungs.
    '''
    edges = roberts(binary)
    binary = ndi.binary_fill_holes(edges)
    if plot == True:
        plots[6].axis('off')
        plots[6].imshow(binary, cmap=plt.cm.bone)
    '''
    Step 8: Superimpose the binary mask on the input image.
    '''
    get_high_vals = binary == 0
    im[get_high_vals] = 0
    if plot == True:
        plots[7].axis('off')
        plots[7].imshow(im, cmap=plt.cm.bone)
    mask = binary.astype(int)

    mask_img = np.zeros(src_img.shape, dtype=src_img.dtype)
    mask_img[mask == 1] = [255, 255, 0]

    if plot == True:
        plt.figure(figsize=(12, 12))
        plt.imshow(mask_img)
        plt.show()

    if mask_img.mean() < 7:
        return (False, [], [], [])
    else:

        # Find the smallest effective rectangle surrounding the lungs
        gray = cv2.cvtColor(mask_img, cv2.COLOR_BGR2GRAY)
        ret, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        contours, hierarchy = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        label_img = np.zeros(src_img.shape)
        cv2.drawContours(label_img, contours, -1, (255, 255, 0), thickness=cv2.FILLED)

        minx, miny, maxx, maxy = src_img.shape[1], src_img.shape[0], 0, 0
        for c in contours:
            if cv2.contourArea(c) > min_lung_area:
                temp_minx = np.min(c[:, 0, 0])
                temp_miny = np.min(c[:, 0, 1])
                temp_maxx = np.max(c[:, 0, 0])
                temp_maxy = np.max(c[:, 0, 1])
                if temp_minx < minx:
                    minx = temp_minx
                if temp_miny < miny:
                    miny = temp_miny
                if temp_maxx > maxx:
                    maxx = temp_maxx
                if temp_maxy > maxy:
                    maxy = temp_maxy
        crop_img = src_img[miny - padding_num:maxy + padding_num, minx - padding_num:maxx + padding_num]
        crop_mask = mask[miny - padding_num:maxy + padding_num, minx - padding_num:maxx + padding_num]
        crop_copy = crop_img.copy()
        crop_img[crop_mask == 0] = [0, 0 , 0]

        if plot == True:
            plt.figure(figsize=(9, 9))
            plt.imshow(crop_img, cmap='gray')
            plt.show()

        # fill crop area
        crop_mask = np.stack([crop_mask] * 3, 2)
        background = entity_filling(crop_img, crop_mask)

        filled_crop_img = np.where(crop_mask == 1, crop_img, background)
        filled_crop_img = np.array(filled_crop_img, dtype=np.int16)

        return (True, filled_crop_img, crop_img, crop_copy)


@jit
def entity_filling(src_img, mask):
    background = np.zeros_like(src_img)
    for i in range(40):
        ix = np.random.randint(0, src_img.shape[0])
        if np.random.randint(2) == 0:
            label_img = np.concatenate(
                [np.zeros((src_img.shape[0] - ix, src_img.shape[1], 3)), src_img[:ix, :]], 0)
            new_mask = np.concatenate(
                [np.zeros((src_img.shape[0] - ix, src_img.shape[1], 3)), mask[:ix, :]], 0)
        else:
            label_img = np.concatenate(
                [src_img[ix:, :], np.zeros((ix, src_img.shape[1], 3))], 0)
            new_mask = np.concatenate(
                [mask[ix:, :], np.zeros((ix, src_img.shape[1], 3))], 0)
        ix = np.random.randint(0, src_img.shape[1])
        if np.random.randint(2) == 0:
            label_img = np.concatenate(
                [np.zeros((src_img.shape[0], src_img.shape[1] - ix, 3)), label_img[:, :ix]], 1)
            new_mask = np.concatenate(
                [np.zeros((src_img.shape[0], src_img.shape[1] - ix, 3)), new_mask[:, :ix]], 1)
        else:
            label_img = np.concatenate(
                [label_img[:, ix:], np.zeros((src_img.shape[0], ix, 3))], 1)
            new_mask = np.concatenate(
                [new_mask[:, ix:], np.zeros((src_img.shape[0], ix, 3))], 1)

        background = np.where(new_mask == 1, label_img, background)
    return background


def make_save(save_path):
    source_dir = os.path.join(save_path, 'source')
    os.makedirs(source_dir, exist_ok=True)
    crop_dir = os.path.join(save_path, 'source_crop')
    os.makedirs(crop_dir, exist_ok=True)
    process_dir = os.path.join(save_path, 'filled')
    os.makedirs(process_dir, exist_ok=True)
    src_crop_dir = os.path.join(save_path, 'crop')
    os.makedirs(src_crop_dir, exist_ok=True)
    return source_dir, crop_dir, process_dir, src_crop_dir


def choose_img(img_list_len, choose_nums=15):
    if img_list_len <= 0:
        return None
    else:
        if img_list_len < 10:
            return np.arange(0, img_list_len)
        elif int(0.5 * img_list_len) < choose_nums:
            choose_nums = int(0.5 * img_list_len)
        return np.arange(int(0.25 * img_list_len), int(0.75 * img_list_len), int(0.5 * img_list_len / choose_nums))


# @jit
# def normalize(img_dirs, sp1, sp2, sp3, crop_imgs, filled_imgs):
#     if sp1!= []:
#         images = []
#         for item in img_dirs:
#             images.append(cv2.imread(item))
#         images = np.array(images)
#         mean = np.mean(images)
#         var = np.mean(np.square(images - mean))
#         images = (images - mean) / np.sqrt(var)
#         for i in range(len(sp1)):
#             image = images[i]
#             crop_img = (crop_imgs[i] - mean) / np.sqrt(var)
#             filled_img = (filled_imgs[i] -mean) / np.sqrt(var)
#             mx = np.max(image)
#             mn = np.min(image)
#             if mx - mn > 0:
#                 image = np.array((image - mn) / (mx - mn) * 255, dtype=np.int16)
#                 crop_img = np.array((crop_img - mn) / (mx - mn) * 255, dtype=np.int16)
#                 filled_img = np.array((filled_img - mn) / (mx - mn) * 255, dtype=np.int16)
#             cv2.imwrite(sp1[i], image)
#             cv2.imwrite(sp2[i], crop_img)
#             cv2.imwrite(sp3[i], filled_img)


def pre_deal(src_img_dir, save_img_dir):
    s1, s2, s3, s4 = make_save(save_img_dir)
    img_list = np.sort(np.array(os.listdir(src_img_dir)))
    choose_index = choose_img(len(img_list))
    for item in img_list[choose_index]:
        img_dir = os.path.join(src_img_dir, item)
        s1_dir = os.path.join(s1, item)
        s2_dir = os.path.join(s2, item)
        s3_dir = os.path.join(s3, item)
        s4_dir = os.path.join(s4, item)
        Flag, filled_crop_img, crop_img, crop_copy = get_segmented_lungs(img_dir)
        if Flag:
            shutil.copy(img_dir, s1_dir)
            cv2.imwrite(s2_dir, crop_copy)
            cv2.imwrite(s3_dir, filled_crop_img)
            cv2.imwrite(s4_dir, crop_img)


if __name__ == '__main__':
    import sys
    input_path, output_path = sys.argv[-2], sys.argv[-1]
    for people in os.listdir(input_path):
        pre_deal(os.path.join(input_path, people), output_path)
