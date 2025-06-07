import warnings
warnings.filterwarnings('ignore')
import argparse
import os
import torch.backends.cudnn as cudnn
from model.unet2d_residual import ResidualUNet2D
from model.Unet_EGNN import unet_egnn
import torchvision.transforms as transforms
# from scipy.ndimage import imread
# from scipy.misc import imsave
import os,sys,csv
from imageio import imread,imsave
from model.loss import *
import time
import random
from glob import glob
import h5py
import pdb
import matplotlib.pyplot as plt
import numpy as np
from train_util import *
from skimage import io
# import sys
# sys.path.append('../cython')
# from connectivity import enforce_connectivity
from sklearn.decomposition import PCA
import h5py,copy
from data.dataset import BBBC
from graph_partition import *
from torch.utils.data import DataLoader
from tqdm import tqdm
from connectivity import enforce_connectivity
# from postprocessing import merge_small_object, merge_func
from data.data_segmentation import relabel
from shutil import copyfile
from postprocessing import merge_small_object, merge_func
from data.data_segmentation import relabel
from cluster import cluster_ms
from utils.show import draw_fragments_2d
from utils.metrics_bbbc import agg_jc_index, pixel_f1, remap_label, get_fast_pq
import skimage
from utils.until import *

from segment_anything import sam_model_registry, SamPredictor
def remove_samll_object(mask, min_size=25):
    annot = skimage.morphology.label(mask)
    annot = skimage.morphology.remove_small_objects(annot, min_size=min_size)
    out_mask = annot.copy()
    out_mask[out_mask != 0] = 1
    return out_mask

def test_epoch(test_loader,visualize_path,model,if_GAEC=False,minsize=10,rd=9):
    if if_GAEC==True:
        visualize_path = visualize_path+'GAEC'
        if not os.path.isdir(visualize_path):
            os.makedirs(visualize_path)
    visualize_path = visualize_path+'/'
    if not os.path.isdir(visualize_path):
        os.makedirs(visualize_path)

    sam_checkpoint = "/data/caimiaomiao/SAM/SAM/segment-anything-main/sam_vit_h_4b8939.pth"
    device = "cuda"
    model_type = "default"
    sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
    sam.to(device=device)
    predictor = SamPredictor(sam)

    aji_score2 = []
    dice_score2 = []
    f1_score2 = []
    pq_score2 = []
    pbar = tqdm(total=len(test_loader))
    for batch_idx, (data, target, fg, boundary, weightmap, affs_mask,imgs_min,imgs_max,images) in enumerate(
            test_loader):  # imgs, label, fg, lb_affs, weightmap, affs_mask

        with torch.no_grad():
            data, target, boundary, weightmap, affs_mask = data.cuda(), target.cuda().long(), \
                                                           boundary.cuda().long(), \
                                                           weightmap.cuda().float(), \
                                                           affs_mask.cuda().float()


            out_boundary, x_emb, binary_seg, edge_feat_list, _, segments, adj = model(data.float(),imgs_min,imgs_max,rd = rd)
            if adj is None:
                continue
            target = target[:, :, 92:-92, 4:-4]
            # spixel loss
            out_boundary = torch.sigmoid(out_boundary)  ###配合下面的这种loss来使用


        out_boundary = out_boundary[0]
        boundary = 1.0 - 0.5 * (out_boundary[0] + out_boundary[1])
        boundary = boundary.cpu().numpy()
        segments = segments[0, 0]


        gt_ins = target[0, 0].cpu().numpy().astype(np.uint16)

        fragments2 = agglo(segments, edge_feat_list, adj).astype(np.uint16)


        x_emb = x_emb.squeeze(0)
        x_emb = np.array(x_emb.cpu())
        shape = x_emb.shape


        pred_mask = F.softmax(binary_seg, dim=1)
        pred_mask = torch.argmax(pred_mask, dim=1).squeeze(0)
        pred_mask_b = pred_mask.data.cpu().numpy()
        pred_mask_b = pred_mask_b.astype(np.uint8)
        pred_mask_b = remove_samll_object(pred_mask_b,min_size=minsize)
        fragments2 = fragments2 * pred_mask_b


        pca = PCA(n_components=3)
        x_emb = np.transpose(x_emb, [1, 2, 0])
        x_emb = x_emb.reshape(-1, 16)
        new_emb = pca.fit_transform(x_emb)
        new_emb = new_emb.reshape(shape[-2], shape[-1], 3)

        # import pdb;pdb.set_trace()

        imgs = images[0].to('cpu').numpy()
        imgs= (imgs / imgs.max()) * 255
        imgs = cv2.cvtColor(imgs.astype(np.uint8), cv2.COLOR_BGR2RGB)
        superpixel=segments
        predict_image=fragments2

        predictor.set_image(imgs)
        new_predict=np.zeros_like(predict_image)
        for index in np.unique(predict_image):
            if index>0:
                predict=(predict_image==index)*255
                target_area = superpixel[predict_image==index]
                superpixel_ids=np.unique(target_area)
                input_points = np.array(calculate_superpixel_centers(superpixel, superpixel_ids))
                input_labels = np.ones(len(input_points))
                input_box=np.array(get_predict_bbox(predict))

                segmentix_instance = Segmentix()
                sam_mask = segmentix_instance.reference_to_sam_mask(predict)

                masks,_, _ = predictor.predict(
                            point_coords=input_points,
                            point_labels=input_labels,
                            box=input_box,
                            mask_input=sam_mask,
                            multimask_output=True,)

                new_predict[masks[0]==True]=index


        if new_predict.max() == 0:
            temp_aji2 = 0.0
            temp_dice2 = 0.0
            temp_f12 = 0.0
            temp_pq2 = 0.0
        else:
            temp_aji2 = agg_jc_index(gt_ins, new_predict)
            temp_f12 = pixel_f1(gt_ins, new_predict)
            gt_relabel = remap_label(gt_ins, by_size=False)
            pred_relabel = remap_label(new_predict, by_size=False)
            pq_info_cur = get_fast_pq(gt_relabel, pred_relabel, match_iou=0.5)[0]
            temp_dice2 = pq_info_cur[0]
            temp_pq2 = pq_info_cur[2]
        aji_score2.append(temp_aji2)
        dice_score2.append(temp_dice2)
        f1_score2.append(temp_f12)
        pq_score2.append(temp_pq2)
        pbar.update(1)
        
        
    pbar.close()

    aji_score_mean2 = np.mean(aji_score2)
    dice_score_mean2 = np.mean(dice_score2)
    f1_score_mean2 = np.mean(f1_score2)
    pq_score_mean2 = np.mean(pq_score2)
    print('aji_score_mean=%.6f dice_score_mean=%.6f f1_score_mean=%.6f pq_score_mean=%.6f' %(aji_score_mean2 , dice_score_mean2 ,f1_score_mean2,pq_score_mean2 ))




def agglo(segments, edge_feat_list, adj):
        ######找到需要merge的i和j##############################################
        layers = edge_feat_list.shape[0]
        # edge_list = edge_feat_list[layers-1,:]
        edge_list = torch.mean(edge_feat_list, 0)
        edge_list = F.softmax(edge_list, dim=0)
        edge_list = torch.argmax(edge_list, 0)
        n_nodes = edge_list.shape[0]
        edge_list = edge_list.float() * adj.float()

        to_merge1 = {}
        for i in range(n_nodes):
            for j in range(i + 1):
                if edge_list[i, j] == 1 and i!=0 and j!=0:
                    to_merge1[(i, j)] = 1
        print('num of merge-pairs:', len(to_merge1.keys()))
        if len(to_merge1.keys())==0:
            return segments
        ######这个循环的作用是建立一个表示超像素如何合并的映射。从大到小，循环会继续追踪，直到找到一个指向自身的超像素。这个过程用于确定超像素的最终合并目标。##############################################
        merge_map = {}
        inverse1, pack1 = np.unique(segments, return_inverse=True)
        pack1 = pack1.reshape(segments.shape)
        labels = copy.deepcopy(inverse1)
        labels = labels.reshape((-1, 1))
        to_merge1 = np.array(list(to_merge1.keys()))
        to_merge1 = np.vstack((to_merge1, np.hstack((labels, labels))))
        for v1, v2 in to_merge1:
            merge_map.setdefault(v1, v1)
            merge_map.setdefault(v2, v2)
            while v1 != merge_map[v1]:
                v1 = merge_map[v1]
            while v2 != merge_map[v2]:
                v2 = merge_map[v2]
            if v1 > v2:
                v1, v2 = v2, v1
            merge_map[v2] = v1
        merge_map[0] = 0
        next_label = 1
        for v in sorted(merge_map.keys()):
            if v == 0:
                continue
            if merge_map[v] == v:
                merge_map[v] = next_label
                next_label += 1
            else:
                merge_map[v] = merge_map[merge_map[v]]

        for idx, val in enumerate(inverse1):
            if val in merge_map:
                val = merge_map[val]  # merge to small id
                inverse1[idx] = val
        segments = inverse1[pack1]
        return segments






def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-b", "--batch_size", type=int, default=1)
    parser.add_argument("-g", "--gpu_nums", type=int, default=1)
    parser.add_argument("-e", "--epochs", type=int, default=1000)
    parser.add_argument("-r", "--lr", type=float, default=1e-4)
    parser.add_argument("-p", "--lr_patience", type=int, default=30)
    parser.add_argument("-n", "--network", type=str, default="unet_egnn(3,[16,32,64,128,256],3,args)")
    parser.add_argument("-t", "--loss_type", type=str, default="BCE_loss")
    parser.add_argument("-d", "--data_dir", type=str, default="/data/caimiaomiao/miccai2024/dataset/BBBC")
    parser.add_argument("-l", "--logs_dir", type=str, default="./log")
    parser.add_argument("-c", "--ckps_dir", type=str, default="./ckp")
    parser.add_argument("-w", "--weight_rate", type=list, default=[10, 1])
    parser.add_argument("-x", "--resume", type=bool, default=False)
    parser.add_argument("-y", "--resume_path", type=str, default=None)
    parser.add_argument("-v", "--visualize_path", type=str, default='./test_visualize_test')

    # spixel
    parser.add_argument('--train_img_height', '-t_imgH', default=256, type=int, help='img height')
    parser.add_argument('--train_img_width', '-t_imgW', default=256, type=int, help='img width')
    parser.add_argument('--input_img_height', '-v_imgH', default=256, type=int, help='img height')
    parser.add_argument('--input_img_width', '-v_imgW', default=256, type=int, help='img width')
    parser.add_argument('--downsize', default=16, type=float, help='grid cell size for superpixel training ')
    parser.add_argument('--pos_weight', '-p_w', default=0.003, type=float, help='weight of the pos term')
    # embedding
    parser.add_argument("-a", "--alpha", type=int, default=1)
    parser.add_argument("-be", "--beta", type=int, default=1)
    parser.add_argument("-ga", "--gama", type=int, default=0.001)
    # EGNN

    # loss rate
    parser.add_argument("-ls", "--loss_spixel", type=int, default=0.02)
    parser.add_argument("-le", "--loss_embedding", type=int, default=1)
    parser.add_argument("-lb", "--loss_binary", type=int, default=6)
    parser.add_argument("-lg", "--loss_gnn", type=int, default=10)

    args = parser.parse_args()
    visualize_path = args.visualize_path
    if not os.path.isdir(visualize_path):
        os.makedirs(visualize_path)

    # create model
    device = torch.device("cuda:0")
    model = eval(args.network)
    model.cuda()
    model.eval()
    valset = BBBC(dir=args.data_dir, mode="validation", size=args.input_img_height)
    testset = BBBC(dir=args.data_dir, mode="test", size=args.input_img_height)
    test_loader = DataLoader(valset,batch_size=1,shuffle=False)
    cudnn.benchmark = True



    rd = 4
    checkpoint = torch.load('./checkpoint.pth',map_location='cuda:0')
    model.load_state_dict(checkpoint['state_dict'])
    test_epoch(test_loader, visualize_path, model,if_GAEC=False,minsize=0,rd =rd)


if __name__ == '__main__':
    main()
