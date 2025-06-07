

from IPython.display import display, HTML
import os
import csv
display(HTML(
"""
<a target="_blank" href="https://colab.research.google.com/github/facebookresearch/segment-anything/blob/main/notebooks/predictor_example.ipynb">
  <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/>
</a>
"""
))

using_colab = False
from shutil import copyfile

import numpy as np
import torch
import matplotlib.pyplot as plt
import cv2

import cv2
import numpy as np
from skimage.measure import label, regionprops
from typing import Tuple
def draw_fragments_2d(pred):
    m,n = pred.shape
    ids = np.unique(pred)
    size = len(ids)
    print("the number of instance is %d" % size)
    color_pred = np.zeros([m, n, 3], dtype=np.uint8)
    idx = np.searchsorted(ids, pred)
    for i in range(3):
        color_val = np.random.randint(0, 255, ids.shape)
        if ids[0] == 0:
            color_val[0] = 0
        color_pred[:,:,i] = color_val[idx]
    return color_pred

def calculate_superpixel_centers(target_area, superpixel_ids):
    """
    计算每个superpixel的中心点坐标。
    """
    centers = []
    for superpixel_id in superpixel_ids:
        # 获取当前superpixel的所有像素坐标
        y_coords, x_coords = np.where(target_area == superpixel_id)

        # 计算中心点坐标
        center_x = int(np.mean(x_coords))
        center_y =  int(np.mean(y_coords))
        centers.append([center_x, center_y])

    return np.array(centers)


def visualize_samples(mask, positive_samples, negative_samples, file_name):
    """
    在mask上可视化正样本点和负样本点，并保存结果。

    :param mask: 二值图像。
    :param positive_samples: 正样本点列表。
    :param negative_samples: 负样本点列表。
    :param file_name: 保存的文件名。
    """
    # 转换mask为彩色图像以便在上面画点
    color_mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)

    # 遍历正样本点和负样本点，分别用绿色和红色标记
    for point, _ in positive_samples:
        cv2.circle(color_mask, np.flip(point), 5, (0, 255, 0), -1)  # 绿色
    for point, _ in negative_samples:
        cv2.circle(color_mask, np.flip(point), 5, (0, 0, 255), -1)  # 红色

    # 保存图像
    cv2.imwrite(file_name, color_mask)

def show_mask(mask, ax, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30/255, 144/255, 255/255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)
    
def show_points(coords, labels, ax, marker_size=375):
    pos_points = coords[labels==1]
    neg_points = coords[labels==0]
    ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)
    ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)   
    
def show_box(box, ax):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0,0,0,0), lw=2))    
def draw_connected_components(image):

    # 连通域标记
    num_labels, labels = cv2.connectedComponents(image)

    # 创建一个与原图像大小相同的彩色图像
    output_image = np.zeros((image.shape[0], image.shape[1], 3), np.uint8)

    # 为每个连通域分配随机颜色
    colors = np.random.randint(0, 255, (num_labels, 3))

    # 绘制连通域
    for label in range(1, num_labels):  # 从 1 开始，忽略背景
        mask = labels == label
        output_image[mask] = colors[label]

    return output_image

def draw_bbox_and_points(image, bbox, interaction_point, label):
    # 如果图像是灰度的，转换为彩色
    if len(image.shape) == 2 or image.shape[2] == 1:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    # 绘制 predict 区域的边界框
    if bbox:
        cv2.rectangle(image, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (255, 0, 0), 2)  # 蓝色框

    # 根据标签在图像上标记交互点
    if interaction_point:
        if label == 0:
            color = (0, 0, 255)  # 正样本点，红色
        else:
            color = (0, 255, 0)  # 负样本点，绿色
        cv2.circle(image, (int(interaction_point[1]), int(interaction_point[0])), 5, color, -1)

    return image
##################################################################


# Function to extract the bounding box of 'predict' area
def get_predict_bbox(predict):
    # 找到所有非零像素的坐标
    coords = cv2.findNonZero(predict)
    if coords is not None:
        # 使用这些坐标计算最小的矩形框
        x, y, w, h = cv2.boundingRect(coords)
        return (x, y, x + w, y + h)
    else:
        return None  # 如果没有非零像素，返回 None

# Function to process fn_map and fp_map, and find the interaction points
def process_maps_and_find_points(fn_map, fp_map,path_save):
    # Thresholding the maps
    threshold = 255 / 2
    fn_thresh = cv2.threshold(fn_map, threshold, 255, cv2.THRESH_BINARY)[1]
    fp_thresh = cv2.threshold(fp_map, threshold, 255, cv2.THRESH_BINARY)[1]

    # 应用函数并保存结果图像
    fn_components = draw_connected_components(fn_thresh)
    fp_components = draw_connected_components(fp_thresh)

    # 保存图像
    cv2.imwrite(path_save+"/fn_components.png", fn_components)
    cv2.imwrite(path_save+"/fp_components.png", fp_components)

    # Finding the largest connected component in both maps
    # 找到两个图像中最大的连通域
    def get_largest_component(binary_map):
        labeled = label(binary_map, connectivity=2)
        regions = regionprops(labeled)
        if not regions:
            return None
        return max(regions, key=lambda r: r.area)

    fn_largest = get_largest_component(fn_thresh)
    fp_largest = get_largest_component(fp_thresh)

    # 比较 fn 和 fp 中最大连通域的大小，选择最大的一个
    if fn_largest and fp_largest:
        if fn_largest.area > fp_largest.area:
            return fn_largest.centroid, 0  # fn_map 中的最大连通域更大，标记为 0
        else:
            return fp_largest.centroid, 1  # fp_map 中的最大连通域更大，标记为 1
    elif fn_largest:
        return fn_largest.centroid, 0
    elif fp_largest:
        return fp_largest.centroid, 1
    else:
        return None, None  # 如果两个图像都没有连通域

    # return fn_center, fn_label, fp_center, fp_label
# Function to read an image from a given path
def load_image(path):
    return cv2.imread(path, cv2.IMREAD_GRAYSCALE)

def extract_samples_from_mask(mask, grid_size, expansion=50):
    """
    从给定的mask中提取正样本点和负样本点。
    
    :param mask: 二值图像，其中前景为255，背景为0。
    :param grid_size: 网格大小。
    :param expansion: 边界框扩展的像素数。
    :return: 正样本点列表、负样本点列表、扩大后的边界框坐标。
    """
    # 创建网格点
    grid_points = [(x, y) for x in range(0, mask.shape[1], grid_size) for y in range(0, mask.shape[0], grid_size)]

    # 提取正样本点和负样本点
    positive_samples = []
    negative_samples = []

    # 计算包含mask的最小边界框
    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)
    xmi, xma = np.where(rows)[0][[0, -1]]
    ymi, yma = np.where(cols)[0][[0, -1]]

    # 扩大边界框
    xmin = max(xmi - expansion, 0)
    xmax = min(xma + expansion, mask.shape[1] - 1)
    ymin = max(ymi - expansion, 0)
    ymax = min(yma + expansion, mask.shape[0] - 1)

    # 计算mask的中心点
    center_x = (xmin + xmax) // 2
    center_y = (ymin + ymax) // 2
    center_point = (center_x, center_y)

    # 判断网格点是否在mask内部或在扩大的边界框内
    for point in grid_points:
        x, y = point
        if mask[x, y] == 255:
            # 正样本点
            positive_samples.append((point, 1))
        elif ymin <= y <= ymax and xmin <= x <= xmax:
            # 负样本点
            negative_samples.append((point, 0))

    return positive_samples, negative_samples, (xmi, ymi, xma, yma),center_point



def extract_box_from_mask(mask):
    """
    从给定的mask中提取正样本点和负样本点。
    
    :param mask: 二值图像，其中前景为255，背景为0。
    :return: 正样本点列表、负样本点列表、扩大后的边界框坐标。
    """


    # 计算包含mask的最小边界框
    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)
    xmi, xma = np.where(rows)[0][[0, -1]]
    ymi, yma = np.where(cols)[0][[0, -1]]

    return (xmi, ymi, xma, yma)
################################################################################################################
from typing import Tuple
import numpy as np

class Segmentix:
    def resize_mask(self, ref_mask: np.ndarray, longest_side: int = 256) -> Tuple[np.ndarray, int, int]:
        """
        Resize an image to have its longest side equal to the specified value.

        Args:
            ref_mask (np.ndarray): The image to be resized.
            longest_side (int, optional): The length of the longest side after resizing. Default is 256.

        Returns:
            tuple[np.ndarray, int, int]: The resized image and its new height and width.
        """
        height, width = ref_mask.shape[:2]
        if height > width:
            new_height = longest_side
            new_width = int(width * (new_height / height))
        else:
            new_width = longest_side
            new_height = int(height * (new_width / width))

        return (
            cv2.resize(
                ref_mask, (new_width, new_height), interpolation=cv2.INTER_NEAREST
            ),
            new_height,
            new_width,
        )

    def pad_mask(
        self,
        ref_mask: np.ndarray,
        new_height: int,
        new_width: int,
        pad_all_sides: bool = False,
    ) -> np.ndarray:
        """
        Add padding to an image to make it square.

        Args:
            ref_mask (np.ndarray): The image to be padded.
            new_height (int): The height of the image after resizing.
            new_width (int): The width of the image after resizing.
            pad_all_sides (bool, optional): Whether to pad all sides of the image equally. If False, padding will be added to the bottom and right sides. Default is False.

        Returns:
            np.ndarray: The padded image.
        """
        pad_height = 256 - new_height
        pad_width = 256 - new_width
        if pad_all_sides:
            padding = (
                (pad_height // 2, pad_height - pad_height // 2),
                (pad_width // 2, pad_width - pad_width // 2),
            )
        else:
            padding = ((0, pad_height), (0, pad_width))

        # Padding value defaults to '0' when the `np.pad`` mode is set to 'constant'.
        return np.pad(ref_mask, padding, mode="constant")

    def reference_to_sam_mask(
        self, ref_mask: np.ndarray, threshold: int = 127, pad_all_sides: bool = False
    ) -> np.ndarray:
        """
        Convert a grayscale mask to a binary mask, resize it to have its longest side equal to 256, and add padding to make it square.

        Args:
            ref_mask (np.ndarray): The grayscale mask to be processed.
            threshold (int, optional): The threshold value for the binarization. Default is 127.
            pad_all_sides (bool, optional): Whether to pad all sides of the image equally. If False, padding will be added to the bottom and right sides. Default is False.

        Returns:
            np.ndarray: The processed binary mask.
        """

        # Convert a grayscale mask to a binary mask.
        # Values over the threshold are set to 1, values below are set to -1.
        ref_mask = np.clip((ref_mask > threshold) * 2 - 1, -1, 1)

        # Resize to have the longest side 256.
        resized_mask, new_height, new_width = self.resize_mask(ref_mask)

        # Add padding to make it square.
        square_mask = self.pad_mask(resized_mask, new_height, new_width, pad_all_sides)

        # Expand SAM mask's dimensions to 1xHxW (1x256x256).
        return np.expand_dims(square_mask, axis=0)
    


import numpy as np

def calculate_iou(gt_mask, pred_mask):
    """
    Calculate the Intersection over Union (IoU) for ground truth and predicted masks.

    Args:
        gt_mask (np.ndarray): The ground truth mask.
        pred_mask (np.ndarray): The predicted mask.

    Returns:
        float: The IoU score.
    """
    # 计算交集
    intersection = np.logical_and(gt_mask, pred_mask)

    # 计算并集
    union = np.logical_or(gt_mask, pred_mask)

    # 计算IoU
    iou_score = np.sum(intersection) / np.sum(union)

    return iou_score
