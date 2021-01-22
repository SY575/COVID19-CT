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
# import shutil
# from tqdm import tqdm
# from numba import jit


def get_segmented_lungs(img_dir, #s1_path, s2_path, s3_path, 
                        plot=False, min_lung_area=4000, padding_num=0):
    """
    :param img_dir: 输入图片路径
    :param s1_path: 原图保存路径
    :param s2_path: 经过截取的有用部分的图片保存路径
    :param s3_path: 经过填充的图片保存路径
    :param plot: 是否展示图片
    :param min_lung_area: 一个肺部的可能最小面积（作为阈值）
    :param padding_num: 截取部分需要padding的大小
    :return: True or False (img is useful or not)
    """
    src_img = cv2.imread(img_dir)
    src_img1 = src_img.copy()
    img = Image.open(img_dir)
    img = np.array(img)[:, :, 0]
    img = img.astype(np.int16)
    if img.mean() < 88 or img.mean() > 133:
        return False, [], []
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

    # plt.figure(figsize=(12, 12))
    # plt.imshow(mask_img)
    # plt.show()

    if mask.mean() < judge_threshold or mask_img.mean() < 6:
        return False, [], []
    # 找出包围肺部的最小有效矩形
    gray = cv2.cvtColor(mask_img, cv2.COLOR_BGR2GRAY)
    ret, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    contours, hierarchy = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    new_img = np.zeros(src_img.shape)
    cv2.drawContours(new_img, contours, -1, (255, 255, 0), thickness=cv2.FILLED)

    # m = new_img[:, :, 0] == 255

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
    final_img = src_img[miny - padding_num:maxy + padding_num, minx - padding_num:maxx + padding_num]
    final_mask = mask[miny - padding_num:maxy + padding_num, minx - padding_num:maxx + padding_num]
    final_img[final_mask == 0] = [0, 0 ,0]
    
    final_crop = src_img1[miny - padding_num:maxy + padding_num, minx - padding_num:maxx + padding_num]
            
    # plt.figure(figsize=(9, 9))
    # plt.imshow(final_img, cmap='gray')
    # plt.show()

    # 填充图片
    final_mask = np.stack([final_mask] * 3, 2)

    background = entity_filling(final_img, final_mask)

    filled_img = np.where(final_mask == 1, final_img, background)
    filled_img = np.array(filled_img, dtype=np.int16)

    # plt.figure(figsize=(12, 12))
    # plt.imshow(background)
    # plt.show()
    return True, final_img, final_crop
# =============================================================================
#     shutil.copy(img_dir, s1_path)
#     cv2.imwrite(s2_path, final_img)
#     cv2.imwrite(s3_path, filled_img)
#     return True
# =============================================================================


# @jit
def entity_filling(src_img, mask):
    background = np.zeros_like(src_img)
    for i in range(40):
        ix = np.random.randint(0, src_img.shape[0])
        if np.random.randint(2) == 0:
            new_img = np.concatenate(
                [np.zeros((src_img.shape[0] - ix, src_img.shape[1], 3)), src_img[:ix, :]], 0)
            new_mask = np.concatenate(
                [np.zeros((src_img.shape[0] - ix, src_img.shape[1], 3)), mask[:ix, :]], 0)
        else:
            new_img = np.concatenate(
                [src_img[ix:, :], np.zeros((ix, src_img.shape[1], 3))], 0)
            new_mask = np.concatenate(
                [mask[ix:, :], np.zeros((ix, src_img.shape[1], 3))], 0)
        ix = np.random.randint(0, src_img.shape[1])
        if np.random.randint(2) == 0:
            new_img = np.concatenate(
                [np.zeros((src_img.shape[0], src_img.shape[1] - ix, 3)), new_img[:, :ix]], 1)
            new_mask = np.concatenate(
                [np.zeros((src_img.shape[0], src_img.shape[1] - ix, 3)), new_mask[:, :ix]], 1)
        else:
            new_img = np.concatenate(
                [new_img[:, ix:], np.zeros((src_img.shape[0], ix, 3))], 1)
            new_mask = np.concatenate(
                [new_mask[:, ix:], np.zeros((src_img.shape[0], ix, 3))], 1)

        background = np.where(new_mask == 1, new_img, background)
    return background


# =============================================================================
# def nor(arr):
#     result = arr.copy()
#     for i in range(arr.shape[0]):
#         MAX = arr[i].max()
#         MIN = arr[i].min()
#         if MAX - MIN != 0:
#             result[i] = (arr[i] - MIN) / (MAX - MIN)
#     result = result * 255
#     result = result.astype(np.uint8)
#     return result
# =============================================================================

def make_save(save_path):
    source_dir = os.path.join(save_path, 'source')
    os.makedirs(source_dir, exist_ok=True)
    crop_dir = os.path.join(save_path, 'crop')
    os.makedirs(crop_dir, exist_ok=True)
    process_dir = os.path.join(save_path, 'filled')
    os.makedirs(process_dir, exist_ok=True)
    return source_dir, crop_dir, process_dir


def choose_img(img_list_len, choose_nums=15):
    a = np.arange(int(0.2 * img_list_len), int(0.8 * img_list_len))
    if len(a) < choose_nums:
        choose_nums = len(a)
    choose_index = np.random.choice(a, choose_nums, replace=False)
    return choose_index

judge_threshold = 0.1125
# =============================================================================
# if __name__ == '__main__':
# 
#     # no_nCov data
#     input_path = r'D:\nCov\raw\no_nCov'
#     output_path = r'D:\nCov\p1\no_nCov'
#     s1, s2, s3 = make_save(output_path)
#     for f in os.listdir(input_path):
#         f1 = os.path.join(input_path, f)
#         img_list = np.sort(np.array(os.listdir(f1)))
#         choose_index = choose_img(len(img_list))
#         for img_name in tqdm(img_list[choose_index]):
#             img_dir = os.path.join(f1, img_name)
#             s1_dir = os.path.join(s1, f + '_' + img_name)
#             s2_dir = os.path.join(s2, f + '_' + img_name)
#             s3_dir = os.path.join(s3, f + '_' + img_name)
#             get_segmented_lungs(img_dir, s1_dir, s2_dir, s3_dir)
# 
#     # first nCov data
#     input_path = r'D:\BaiduNetdiskDownload\2019-nCoV_jpg'
#     output_path = r'D:\nCov\p1\sysu_san'
#     s1, s2, s3 = make_save(output_path)
#     for f in os.listdir(input_path):
#         f1 = os.path.join(input_path, f)
#         for f2 in os.listdir(f1):
#             f3 = os.path.join(f1, f2)
#             for f4 in os.listdir(f3):
#                 f5 = os.path.join(f3, f4)
#                 img_list = np.sort(np.array(os.listdir(f5)))
#                 choose_index = choose_img(len(img_list))
#                 for img_name in tqdm(img_list[choose_index]):
#                     img_dir = os.path.join(f5, img_name)
#                     s1_dir = os.path.join(s1, f + '_' + img_name)
#                     s2_dir = os.path.join(s2, f + '_' + img_name)
#                     s3_dir = os.path.join(s3, f + '_' + img_name)
#                     get_segmented_lungs(img_dir, s1_dir, s2_dir, s3_dir)
# 
#     # wuhan nCov data
#     input_path = r'D:\BaiduNetdiskDownload\wh_jpg\wh_jpg'
#     output_path = r'D:\nCov\p1\wuhan'
#     s1, s2, s3 = make_save(output_path)
#     for img_name in tqdm(os.listdir(input_path)):
#         img_dir = os.path.join(input_path, img_name)
#         s1_dir = os.path.join(s1, img_name)
#         s2_dir = os.path.join(s2, img_name)
#         s3_dir = os.path.join(s3, img_name)
#         get_segmented_lungs(img_dir, s1_dir, s2_dir, s3_dir)
# 
#     # germ data
#     input_path = r'D:\BaiduNetdiskDownload\germ_cjw\germ'
#     output_path = r'D:\nCov\p1\germ'
#     s1, s2, s3 = make_save(output_path)
#     for img_name in tqdm(os.listdir(input_path)):
#         img_dir = os.path.join(input_path, img_name)
#         s1_dir = os.path.join(s1, img_name)
#         s2_dir = os.path.join(s2, img_name)
#         s3_dir = os.path.join(s3, img_name)
#         get_segmented_lungs(img_dir, s1_dir, s2_dir, s3_dir)
# 
#     # 2-9 data
#     input_path = r'D:\BaiduNetdiskDownload\new'
#     output_path = r'D:\nCov\p1\wuhan'
#     s1, s2, s3 = make_save(output_path)
#     for sub in os.listdir(input_path):
#         fn = os.path.join(input_path, sub)
#         for f in os.listdir(fn):
#             f1 = os.path.join(fn, f)
#             for f2 in os.listdir(f1):
#                 f3 = os.path.join(f1, f2)
#                 img_list = os.listdir(f3)
#                 if img_list[0][-4:] == '.jpg':
#                     img_list = np.sort(np.array(img_list))
#                     choose_index = choose_img(len(img_list))
#                     for img_name in tqdm(img_list[choose_index]):
#                         img_dir = os.path.join(f3, img_name)
#                         s1_dir = os.path.join(s1, f + '_' + img_name)
#                         s2_dir = os.path.join(s2, f + '_' + img_name)
#                         s3_dir = os.path.join(s3, f + '_' + img_name)
#                         get_segmented_lungs(img_dir, s1_dir, s2_dir, s3_dir)
# 
#     # 2-10 no_nCov
#     input_path = r'D:\Common_no_nCov'
#     output_path = r'D:\nCov\p1\2_10_no_nCov'
#     s1, s2, s3 = make_save(output_path)
#     img_list = os.listdir(input_path)
#     patients = set([item.split('_')[0] for item in img_list])
#     pat_dict = {}
#     for p in patients:
#         pat_dict[p] =[]
#     for ijk in img_list:
#         pat_dict[ijk.split('_')[0]].append(ijk)
#     for key in pat_dict.keys():
#         choose_index = choose_img(len(pat_dict[key]))
#         img_lst = np.sort(np.array(pat_dict[key]))
#         for img_name in tqdm(img_lst[choose_index]):
#             img_dir = os.path.join(input_path, img_name)
#             s1_dir = os.path.join(s1, img_name)
#             s2_dir = os.path.join(s2, img_name)
#             s3_dir = os.path.join(s3, img_name)
#             get_segmented_lungs(img_dir, s1_dir, s2_dir, s3_dir)
# =============================================================================
