# -*- coding: utf-8 -*-
# @Time : 2021/3/30
# @Author : Xiaoyu Liu
# @Email : liuxyu@mail.ustc.edu.cn
# @Software: PyCharm

from pathlib import Path
import SimpleITK as sitk
from skimage.segmentation import slic, mark_boundaries
from skimage import io
import numpy as np
from tqdm import tqdm
from scipy.ndimage import binary_erosion,binary_dilation,binary_opening
import torch
import torch.nn as nn
import torch.nn.functional as F
import networkx as nx
import matplotlib.pyplot as plt
import cv2

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

def ensure_dir(dirname):
    dirname = Path(dirname)
    if not dirname.is_dir():
        dirname.mkdir(parents=True, exist_ok=False)

def log_args(args,log):
    args_info = "\n##############\n"
    for key in args.__dict__:
        args_info = args_info+(key+":").ljust(25)+str(args.__dict__[key])+"\n"
    args_info += "##############"
    log.info(args_info)

def save_nii(img,save_name):
    nii_image = sitk.GetImageFromArray(img)
    name = str(save_name).split("/")
    sitk.WriteImage(nii_image,str(save_name))
    print(name[-1]+" saving finished!")


def get_graph_from_image(segments,embedding): #Tensor
    """
    segments: HxW
    embedding: FxHxW

    #猝： 很多麻烦的东西，构建这样一个图只是用torch
    """

    NP_TORCH_FLOAT_DTYPE = torch.float32
    NP_TORCH_LONG_DTYPE = torch.int64
    NUM_FEATURES = embedding.shape[0]+len(segments.shape)
    # load the segments and convert it to rag

    num_nodes = np.max(segments)
    # nodes = {
    #     node: {
    #         "emb_list": [],
    #         "pos_list": []
    #     } for node in range(num_nodes + 1)
    # }

    height = segments.shape[0]
    width = segments.shape[1]
    # for y in range(height):
    #     for x in range(width):
    #         node = segments[y, x]
    #         emb = embedding[:,y, x]
    #         pos = torch.tensor([float(x) / width, float(y) / height]).cuda()
    #         nodes[node.item()]["emb_list"].append(emb)
    #         nodes[node.item()]["pos_list"].append(pos)
        # end for
    # end for

    # G = nx.Graph()
    #
    # for node in nodes:
    #     nodes[node]["emb_list"] = torch.stack(nodes[node]["emb_list"])
    #     nodes[node]["pos_list"] = torch.stack(nodes[node]["pos_list"])
    #     # emb
    #     emb_mean = torch.mean(nodes[node]["emb_list"], dim=0)
    #     # rgb_std = np.std(nodes[node]["rgb_list"], axis=0)
    #     # rgb_gram = np.matmul( nodes[node]["rgb_list"].T, nodes[node]["rgb_list"] ) / nodes[node]["rgb_list"].shape[0]
    #     # Pos
    #     pos_mean = torch.mean(nodes[node]["pos_list"], dim=0)
    #     # pos_std = np.std(nodes[node]["pos_list"], axis=0)
    #     # pos_gram = np.matmul( nodes[node]["pos_list"].T, nodes[node]["pos_list"] ) / nodes[node]["pos_list"].shape[0]
    #     # Debug
    #
    #     features = torch.cat(
    #         [
    #             torch.reshape(emb_mean, (-1,)),
    #             # np.reshape(rgb_std, -1),
    #             # np.reshape(rgb_gram, -1),
    #             torch.reshape(pos_mean, (-1,)),
    #             # np.reshape(pos_std, -1),
    #             # np.reshape(pos_gram, -1)
    #         ]
    #     )
    #     G.add_node(node, features=list(features))
    G = nx.Graph()
    #计算每个超像素内所有像素的特征嵌入的均值##########################################################
    for node in range(num_nodes + 1):
        mask = segments==node
        mask_erode = binary_erosion(mask, structure=np.ones((3, 3)), iterations=2, border_value=True)
        if np.sum(mask_erode)<5:
            mask_erode = mask
        mask_erode = torch.tensor(mask_erode.astype(np.bool))
        # import pdb
        # pdb.set_trace()
        emb_mean = torch.mean(embedding[:,mask_erode],dim=1)#计算每个超像素内所有像素的特征嵌入的均值
        pos_y,pos_x = np.where(mask)
        pos_y, pos_x = np.mean(pos_y)/height, np.mean(pos_x)/width
        pos_mean = torch.tensor([pos_y,pos_x]).cuda()
        features = torch.cat(
            [
                torch.reshape(emb_mean, (-1,)),
                # np.reshape(rgb_std, -1),
                # np.reshape(rgb_gram, -1),
                torch.reshape(pos_mean, (-1,)),
                # np.reshape(pos_std, -1),
                # np.reshape(pos_gram, -1)
            ]
        )
        G.add_node(node, features=list(features))
    ##########################################################
    # end

    # From https://stackoverflow.com/questions/26237580/skimage-slic-getting-neighbouring-segments
    # segments_ids = np.unique(segments)
    #
    # # centers
    # centers = np.array([np.mean(np.nonzero(segments == i), axis=1) for i in segments_ids])
    # vs_right = torch.cat([segments[:, :-1].flatten().unsqueeze(0), segments[:, 1:].flatten().unsqueeze(0)],dim=0)
    # vs_below = torch.cat([segments[:-1, :].flatten().unsqueeze(0), segments[1:, :].flatten().unsqueeze(0)],dim=0)
    #######################判断超像素是否相邻##########################################################################
    vs_right = np.vstack([segments[:,:-1].ravel(), segments[:,1:].ravel()])  #vs_right 和 vs_below 通过比较像素与其右侧和下方的像素来创建两个数组
    vs_below = np.vstack([segments[:-1,:].ravel(), segments[1:,:].ravel()])  

    bneighbors = np.unique(np.hstack([vs_right, vs_below]), axis=1)   #np.hstack 将 vs_right 和 vs_below 水平堆叠，形成一个更大的矩阵，其中包含了所有可能的邻接像素对。np.unique 用于移除重复的像素对，并确保每个邻接关系只被计算一次

    # Adjacency loops  通过遍历 bneighbors 中的每一对像素，如果两个像素属于不同的超像素（即 bneighbors[0, i] != bneighbors[1, i]），则在它们对应的超像素之间添加一条边。这表明这两个超像素在图像中是物理上相邻的。
    for i in range(bneighbors.shape[1]):
        if bneighbors[0, i] != bneighbors[1, i]:
            G.add_edge(bneighbors[0, i].item(), bneighbors[1, i].item())
    ##################################################################################################################

    # Self loops
    for node in range(num_nodes + 1):
        G.add_edge(node, node)

    n = len(G.nodes)
    m = len(G.edges)
    h = torch.zeros([n, NUM_FEATURES]).type(NP_TORCH_FLOAT_DTYPE)
    edges = torch.zeros([2 * m, 2]).type(NP_TORCH_LONG_DTYPE)
    for e, (s, t) in enumerate(G.edges):
        edges[e, 0] = s
        edges[e, 1] = t

        edges[m + e, 0] = t
        edges[m + e, 1] = s
    # end for
    for i in G.nodes:
        for f in range(NUM_FEATURES):
            h[i, f] = G.nodes[i]["features"][f]  # grad is false????
    # end for
    del G
    return h, edges #节点和边

def batch_graphs(gs): #input batch-size graphs from function:get_graph_from_image
    """
    Assure that every different graph have no identical IDS.
    """
    NP_TORCH_FLOAT_DTYPE = torch.float32
    NP_TORCH_LONG_DTYPE = torch.int64
    NUM_FEATURES = gs[0][0].shape[-1] #a node
    G = len(gs) #batch_size
    N = sum(g[0].shape[0] for g in gs) #number of all nodes from different graph
    M = sum(g[1].shape[0] for g in gs) #number of all relations-edges from different graph
    adj = torch.zeros([N, N]) # big adjacent matrix
    src = torch.zeros([M])
    tgt = torch.zeros([M])
    Msrc = torch.zeros([N, M])
    Mtgt = torch.zeros([N, M])
    Mgraph = torch.zeros([N, G])
    h = torch.cat([g[0] for g in gs]) #all nodes

    n_acc = 0
    m_acc = 0
    for g_idx, g in enumerate(gs):
        n = g[0].shape[0] #number of  nodes from one graph
        m = g[1].shape[0] #number of  edegs from one graph

        for e, (s, t) in enumerate(g[1]): #edges from one graph
            adj[n_acc + s, n_acc + t] = 1
            adj[n_acc + t, n_acc + s] = 1

            src[m_acc + e] = n_acc + s #node1
            tgt[m_acc + e] = n_acc + t #node2

            Msrc[n_acc + s, m_acc + e] = 1 #node1-nodes(including many repeated nodes)
            Mtgt[n_acc + t, m_acc + e] = 1 #node2-nodes

        Mgraph[n_acc:n_acc + n, g_idx] = 1

        n_acc += n
        m_acc += m
    return (
        h.astype(NP_TORCH_FLOAT_DTYPE),
        adj.astype(NP_TORCH_FLOAT_DTYPE),
        src.astype(NP_TORCH_LONG_DTYPE),
        tgt.astype(NP_TORCH_LONG_DTYPE),
        Msrc.astype(NP_TORCH_FLOAT_DTYPE),
        Mtgt.astype(NP_TORCH_FLOAT_DTYPE),
        Mgraph.astype(NP_TORCH_FLOAT_DTYPE)
    )



def Segments2RAG(segments,embedding):
    "segments: BxHxW"
    "embedding: BxCxHxW"
    NP_TORCH_FLOAT_DTYPE = np.float32
    NP_TORCH_LONG_DTYPE = np.int64
    NUM_FEATURES = embedding.shape[1] + +len(segments.shape)-1
    Batch_size = embedding.shape[0]

    #convert to new id without overlap between different Batch
    max_id_list = [segments[b].max() for b in range(Batch_size)]
    for b in range(Batch_size):
        if b>0:
            segments[b] = segments[b]+max_id_list[b-1]
            segments[b][segments[b]==max_id_list[b-1]]=0

    ## load one segments and convert it to one rag
    # for b in range(Batch_size):
    #     g = get_graph_from_image(segments[b],embedding[b])  创建单个图的列表
    gs = [get_graph_from_image(segments[b],embedding[b])  for b in range(Batch_size)] #graph list
    G = len(gs)  # batch_size
    N = sum(g[0].shape[0] for g in gs) #number of all nodes from different graph
    adj = torch.zeros([N, N]).cuda() # big adjacent matrix
    h = torch.cat([g[0] for g in gs]).cuda() #all nodes
    n_acc = 0

    for g_idx, g in enumerate(gs):
        for e, (s, t) in enumerate(g[1]):  # edges from one graph
            adj[n_acc + s, n_acc + t] = 1
            adj[n_acc + t, n_acc + s] = 1

    return h,adj




def compute_cosine_embedding(node_feat):
    """计算所有超像素特征之间的余弦相似度矩阵，使用 PyTorch 张量"""
    n = node_feat.size(0)  # 超像素的数量
    cosine_sim_matrix = torch.zeros((n, n), device=node_feat.device)  # 使用与 node_feat 相同的设备

    for i in range(n):
        for j in range(n):
            vec_a = node_feat[i]
            vec_b = node_feat[j]
            cosine_sim_matrix[i, j] = torch.nn.functional.cosine_similarity(vec_a.unsqueeze(0), vec_b.unsqueeze(0))

    return cosine_sim_matrix

def custom_cosine_similarity(vec_a, vec_b):
    """计算两个向量之间的余弦相似度"""
    dot_product = torch.dot(vec_a, vec_b)
    norm_a = torch.norm(vec_a)
    norm_b = torch.norm(vec_b)
    return dot_product / (norm_a * norm_b) if norm_a * norm_b != 0 else 0

def compute_cosine_similarity_histogram(segments, adj,x, num_bins=256):
    """根据segments和adj计算余弦相似度矩阵，使用 PyTorch 张量"""
    """计算每个超像素的直方图，使用 PyTorch 张量"""
    
    n_nodes = adj.shape[0]

    device = adj.device
    histograms = torch.zeros((n_nodes, num_bins), device=device)
    # 计算每个超像素的直方图
    for i in range(n_nodes):
        superpixel_mask = (segments == i)
        hist = torch.histc((x[superpixel_mask[0,0]]).float().cpu(), bins=num_bins, min=0, max=num_bins)
        histograms[i, :] = hist

    # 归一化直方图以便于计算余弦相似度
    norms = torch.norm(histograms, p=2, dim=1, keepdim=True)
    normalized_histograms = histograms / norms

    cosine_sim_matrix = torch.zeros_like(adj, dtype=torch.float32, device=device)
    for i in range(n_nodes):
        for j in range(i + 1):
            if adj[i, j] == 1:
                vec_a = normalized_histograms[i]
                vec_b = normalized_histograms[j]
                cosine_similarity = custom_cosine_similarity(vec_a, vec_b)
                cosine_sim_matrix[i, j] = cosine_similarity
                cosine_sim_matrix[j, i] = cosine_similarity  # 对称矩阵

    return cosine_sim_matrix

def calculate_centers(superpixel_mask):
    """
    计算每个superpixel的中心点坐标。
    """
    centers = []
    # 获取当前superpixel的所有像素坐标
    y_coords, x_coords = np.where(superpixel_mask)

    # 计算中心点坐标
    center_x = int(np.mean(x_coords))
    center_y =  int(np.mean(y_coords))
    centers.append([center_x, center_y])

    return np.array(centers)

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
    iou_score = np.sum(intersection) / (np.sum(union)+1e-6)

    return iou_score


def compute_SAM_iou(segments, adj,x,predictor):
    """根据segments和adj计算余弦相似度矩阵，使用 PyTorch 张量"""
    """计算每个超像素的直方图，使用 PyTorch 张量"""
    
    n_nodes = adj.shape[0]

    device = adj.device
    SAM_mask = []
    # 计算每个超像素的SAMmask
    for i in range(n_nodes):
        superpixel_mask = (segments[0,0] == i)
        # predictor.set_image(x)
        input_points = np.array(calculate_centers(superpixel_mask))
        input_labels = np.ones(len(input_points))
        masks,scores, logits = predictor.predict(
            point_coords=input_points,
            point_labels=input_labels,
            multimask_output=True,)
        SAM_mask.append(masks[0])

        # plt.figure(figsize=(10,10))
        # plt.imshow(x)
        # show_mask(masks[0], plt.gca())
        # # show_box(input_box, plt.gca())
        # show_points(input_points, input_labels, plt.gca())
        # plt.axis('off')
        # plt.savefig('/data/caimiaomiao/BISSEG/BISSG/BBBC/V4_cov_32/fig/{}.png'.format(i))
        # plt.show() 

        cv2.imwrite('/data/caimiaomiao/BISSEG/BISSG/BBBC/V4_cov_32/fig/{}_super.png'.format(i),superpixel_mask*255)
# for point, 
        
    # 归一化
    # norms = torch.norm(SAM_iou, p=2, dim=1, keepdim=True)
    # normalized_SAM_iou = SAM_iou / norms

    sam_iou_matrix = torch.zeros_like(adj, dtype=torch.float32, device=device)
    for i in range(n_nodes):
        for j in range(i + 1):
            if adj[i, j] == 1:
                vec_a = SAM_mask[i]
                vec_b = SAM_mask[j]
                sam_iou = calculate_iou(vec_a, vec_b)
                sam_iou_matrix[i, j] = sam_iou
                sam_iou_matrix[j, i] = sam_iou  # 对称矩阵

    return sam_iou_matrix



if __name__ == "__main__":
    import h5py
    import torch
    spixel_path = r'C:\Users\Mr.Liu\Desktop\Code_survey\Code_spix_embedding2\outputs\ID\0000.tif'
    segments = io.imread(spixel_path)
    # segments = torch.tensor(segments.astype(np.int64))
    segments = segments.astype(np.int64)
    emb_path = r'C:\Users\Mr.Liu\Desktop\Code_survey\Code_spix_embedding2\outputs\embedding\0001.hdf'
    with h5py.File(emb_path,'r') as f:
        embedding = f['main'][:]
    embedding = torch.tensor(embedding).cuda()
    print(embedding.shape)
    # graph = get_graph_from_image(segments,embedding)
    graph = Segments2RAG(segments[np.newaxis,...],embedding.unsqueeze(0))
    #graph = Segments2RAG(torch.cat((segments.unsqueeze(0),segments.unsqueeze(0))), torch.cat((embedding.unsqueeze(0),embedding.unsqueeze(0))))
    print(graph[0].shape,graph[0].dtype,graph[1].shape,graph[1].dtype)


    # 使用示例
# segments = ... # 2D 分割图
# adj = ... # 邻接矩阵
# cos_sim_matrix = compute_cosine_similarity_from_segments_adj(segments, adj)
# cos_sim_matrix[i, j] 将给出第 i 个和第 j 个超像素之间的关系


