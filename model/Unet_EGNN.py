import torch
from torch import nn
import torch.nn.functional as F
from .unet2d_residual import ResidualUNet2D
from utils.utils_rag_matrix import *
from .GCNN_model import *
import numpy as np
from train_util import *
import time
sys.path.append('../third_party/cython')
from connectivity import enforce_connectivity
from utils.fragment import watershed, randomlabel
from utils.water import *
import cython_utils
from postprocessing import merge_small_object, merge_func, remove_samll_object
from data.data_segmentation import relabel


class unet_egnn(nn.Module):
    def __init__(self, n_channels,nfeatures,graph_layers,args,n_emb=16,n_pos=9):
        super(unet_egnn, self).__init__()
        self.in_channels = n_channels
        self.n_features = nfeatures
        self.graph_layers = graph_layers
        self.args = args
        self.n_emb = n_emb
        self.n_pos = n_pos
        self.unet = ResidualUNet2D(self.in_channels,self.n_features)
        for p in self.parameters():
            p.requires_grad = False 
        self.egnn = GraphNetwork(16+32+32+2,32+2,32+2,self.graph_layers,dropout=0.5)
        sam_checkpoint = "/data/caimiaomiao/SAM/SAM/segment-anything-main/sam_vit_h_4b8939.pth"
        device = "cuda"
        model_type = "default"
        import sys
        sys.path.append("..")
        from segment_anything import sam_model_registry, SamPredictor
        sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
        sam.to(device=device)
        predictor = SamPredictor(sam)
        self.predictor=predictor

        for param in self.predictor.model.parameters():
            param.requires_grad = False


        self.upsampling_model = nn.Sequential(
            nn.ConvTranspose2d(256, 256, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(256, 256, kernel_size=4, stride=2, padding=1), 
            nn.ReLU()
        )

        self.conv = nn.Conv2d(256, 32, kernel_size=1).cuda()


    def forward(self, x,imgs_min,imgs_max,mode='validation',rd=8):
        out_boundary, out_emb, out_binary_seg,spix4  = self.unet(x) 
        boundary = torch.sigmoid(out_boundary.detach())
        boundary = boundary[0]
        boundary = 1.0 - 0.5 * (boundary[0] + boundary[1])
        boundary = boundary.cpu().numpy()


        pred_mask = F.softmax(out_binary_seg, dim=1)
        pred_mask = torch.argmax(pred_mask, dim=1).squeeze(0)
        pred_mask_b = pred_mask.data.cpu().numpy()
        pred_mask_b = pred_mask_b.astype(np.uint8)

        segments = gen_fragment(boundary * (pred_mask_b == 0), radius=rd)
        if mode == 'validation':
            segments = segments.astype(np.uint16)[92:-92, 4:-4]
            pred_mask_b = pred_mask_b[92:-92, 4:-4]
        else:
            segments = segments.astype(np.uint16)
        pred_mask_b = remove_samll_object(pred_mask_b)
        segments = segments * pred_mask_b
        if mode=='validation':
            out_emb = out_emb[:,:,92:-92, 4:-4]
            spix4 = spix4[:,:,92:-92,4:-4]
            out_binary_seg = out_binary_seg[:,:,92:-92,4:-4]
            boundary = boundary[92:-92,4:-4]
            x=x[:,:,92:-92,4:-4]
        inverse1, pack1,counts = np.unique(segments, return_inverse=True,return_counts=True)
        #print(len(inverse1))
        if len(inverse1) <= 1:
            return out_boundary,out_emb,out_binary_seg,None,None,segments,None

        # print('IDs:',len(np.unique(segments)))
        inverse1, pack1= np.unique(segments, return_inverse=True)
        segments = segments[np.newaxis,...]
        segments = segments[np.newaxis, ...]

        pack1 = pack1.reshape(segments.shape)
        inverse1 = np.arange(0, len(inverse1))
        segments = inverse1[pack1]

        time0 = time.time()
        
        sam_x=x[0,0]* (imgs_max.cuda() - imgs_min.cuda()) + imgs_min.cuda()
        sam_x= (sam_x / sam_x.max()) * 255
        sam_x = cv2.cvtColor(sam_x.cpu().numpy().astype(np.uint8), cv2.COLOR_BGR2RGB)
        self.predictor.set_image(sam_x)
        image_embedding = self.predictor.get_image_embedding()
        SAM_embedding=F.interpolate(image_embedding, size=(x.shape[2], x.shape[3]), mode='bicubic', align_corners=False)
        SAM_embedding=self.conv(SAM_embedding)
        node_feat, adj = Segments2RAG(segments[:,0,:],torch.cat((out_emb,spix4,SAM_embedding),1)) 
        
        adj_num,adj_intensity = cython_utils.get_adj(segments[0,0],adj.cpu().numpy().astype(np.int64),boundary) 
        adj_num = adj_num/adj_num.max() 
        adj_intensity = adj_intensity/255 

        adj_num = torch.tensor(adj_num).cuda().unsqueeze(0)
        adj_intensity = torch.tensor(adj_intensity).cuda().unsqueeze(0)
        adj_boundary = torch.cat((adj_intensity,adj_num)).float() 
        time1 = time.time()
        print('Segments2RAG Time:',time1-time0)
        adj_b = adj * (1 - adj_boundary[0])
        edge_feat = adj_b + 0
        edge_feat_neg = adj * (adj_boundary[0])
        edge_feat_all = torch.cat((edge_feat_neg.unsqueeze(0),edge_feat.unsqueeze(0))) #计算边特征

        sam_img=torch.from_numpy(sam_x[:,:,0]).float().cuda()
        cos_aff=compute_cosine_similarity_histogram(torch.from_numpy(segments).cuda(), adj,sam_img).float()
        SAM_iou=compute_SAM_iou(segments, adj,sam_x,self.predictor).float()

        edge_feat_list, node_feat_list = self.egnn(node_feat.unsqueeze(0),edge_feat_all.unsqueeze(0),adj,adj_boundary.unsqueeze(0),cos_aff.unsqueeze(0),SAM_iou.unsqueeze(0)) #通过EGNN处理节点和边特征: [31, 50]

        edge_feat_list = torch.cat(edge_feat_list)
        node_feat_list = torch.cat(node_feat_list)

        return out_boundary,out_emb,out_binary_seg,edge_feat_list,node_feat_list,segments,adj


