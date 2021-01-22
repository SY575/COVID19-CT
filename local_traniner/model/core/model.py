from torch import nn
import torch
import torch.nn.functional as F
from torch.autograd import Variable
from core import resnet
import numpy as np
from core.anchors import generate_default_anchor_maps, hard_nms, cdds2attention_map
from config import CAT_NUM, PROPOSAL_NUM
import torchvision
from torchvision import transforms
from PIL import Image
import torch.nn.functional as F
import numpy as np
from torch.nn.modules.utils import _pair

class ProposalNet(nn.Module):
    def __init__(self):
        super(ProposalNet, self).__init__()
        self.down1 = nn.Conv2d(2048, 128, 3, 1, 1)
        self.down2 = nn.Conv2d(128, 128, 3, 2, 1)
        self.down3 = nn.Conv2d(128, 128, 3, 2, 1)
        self.ReLU = nn.ReLU()
        self.tidy1 = nn.Conv2d(128, 1, 1, 1, 0)
        self.tidy2 = nn.Conv2d(128, 1, 1, 1, 0)
        self.tidy3 = nn.Conv2d(128, 1, 1, 1, 0)

    def forward(self, x):
        batch_size = x.size(0)
        d1 = self.ReLU(self.down1(x))
        d2 = self.ReLU(self.down2(d1))
        d3 = self.ReLU(self.down3(d2))
        t1 = self.tidy1(d1).view(batch_size, -1)
        t2 = self.tidy2(d2).view(batch_size, -1)
        t3 = self.tidy3(d3).view(batch_size, -1)
        return torch.cat((t1, t2), dim=1), t3


class attention_net(nn.Module):
    def __init__(self, topN=4, n_class=200):
        super(attention_net, self).__init__()
        self.n_class = n_class
        self.pretrained_model = resnet.resnet50(pretrained=True)
        self.pretrained_model.avgpool = nn.AdaptiveAvgPool2d(1)
        self.pretrained_model.fc = nn.Linear(512 * 4, self.n_class)
        self.proposal_net = ProposalNet()
        self.topN = topN
        self.concat_net = nn.Linear(2048 * (CAT_NUM + 1 + 1), self.n_class)
        self.partcls_net = nn.Linear(512 * 4, self.n_class)
        
        self.pad_side = 224
        _, edge_anchors_small, _ = generate_default_anchor_maps(setting='small')
        self.edge_anchors_small = (edge_anchors_small + 224).astype(np.int)
        _, edge_anchors_large, _ = generate_default_anchor_maps(setting='large')
        self.edge_anchors_large = (edge_anchors_large + 224).astype(np.int)
        
        
        

    def forward(self, x, img_raw, add=False, return_vis=False):
        resnet_out, rpn_feature, feature = self.pretrained_model(x)
        x_pad = F.pad(x, (self.pad_side, self.pad_side, 
                          self.pad_side, self.pad_side), mode='constant', value=0)
# =============================================================================
#         np.save('./x_pad.npy', x_pad.data.cpu().numpy())
#         np.save('./x.npy', x.data.cpu().numpy())
#         assert 0
# =============================================================================
        batch = x.size(0)
        # small
        rpn_score_small, rpn_score_large = self.proposal_net(rpn_feature.detach())
        all_cdds_small = [
            np.concatenate((x.reshape(-1, 1), 
                            self.edge_anchors_small.copy(), 
                            np.arange(0, len(x)).reshape(-1, 1)), axis=1)
            for x in rpn_score_small.data.cpu().numpy()]
        top_n_cdds_small = [hard_nms(x, topn=self.topN//2, iou_thresh=0.1) for x in all_cdds_small]
        top_n_cdds_small = np.array(top_n_cdds_small)
        top_n_index_small = top_n_cdds_small[:, :, -1].astype(np.int)
        top_n_index_small = torch.from_numpy(top_n_index_small).cuda()
        top_n_prob_small = torch.gather(rpn_score_small, dim=1, index=top_n_index_small)
        # large
        rpn_score_large, rpn_score_large = self.proposal_net(rpn_feature.detach())
        all_cdds_large = [
            np.concatenate((x.reshape(-1, 1), 
                            self.edge_anchors_large.copy(), 
                            np.arange(0, len(x)).reshape(-1, 1)), axis=1)
            for x in rpn_score_large.data.cpu().numpy()]
        top_n_cdds_large = [hard_nms(x, topn=self.topN//2, iou_thresh=0.1) for x in all_cdds_large]
        top_n_cdds_large = np.array(top_n_cdds_large)
        top_n_index_large = top_n_cdds_large[:, :, -1].astype(np.int)
        top_n_index_large = torch.from_numpy(top_n_index_large).cuda()
        top_n_prob_large = torch.gather(rpn_score_large, dim=1, index=top_n_index_large)
        
        
        part_imgs = torch.zeros([batch, self.topN, 3, 224, 224]).cuda()
        for i in range(batch):
            for j in range(self.topN//2):
                [y0, x0, y1, x1] = top_n_cdds_small[i][j, 1:5].astype(np.int)
                part_imgs[i:i + 1, j] = F.interpolate(x_pad[i:i + 1, :, y0:y1, x0:x1], size=(224, 224), mode='bilinear',
                                                      align_corners=True)
                [y0, x0, y1, x1] = top_n_cdds_large[i][j, 1:5].astype(np.int)
                part_imgs[i:i + 1, j+self.topN//2] = F.interpolate(x_pad[i:i + 1, :, y0:y1, x0:x1], size=(224, 224), mode='bilinear',
                                                      align_corners=True)
                
        part_imgs = part_imgs.view(batch * self.topN, 3, 224, 224)
        temp, _, part_features = self.pretrained_model(part_imgs.detach())
        part_feature = part_features.view(batch, self.topN, -1)
        part_feature = part_feature[:, :CAT_NUM, ...].contiguous()
        part_feature = part_feature.view(batch, -1)
# =============================================================================
#         
# =============================================================================
        x2 = x.clone()
        if add:
            for bs in range(batch):
                [y0, x0, y1, x1] = top_n_cdds_large[bs][0, 1:5].astype(np.int)
                y0, x0, y1, x1 = get_xy(y0, x0, y1, x1)
                y0 = np.int((y0 - 224)/448*600)
                x0 = np.int((x0 - 224)/448*600)
                y1 = np.int((y1 - 224)/448*600)
                x1 = np.int((x1 - 224)/448*600)
                x2[bs] = F.interpolate(
                        img_raw[bs:bs + 1, :, y0:y1, x0:x1],
                        size=(448, 448), mode='bilinear', align_corners=True)
        _, _, feature2 = self.pretrained_model(x2.detach()) # 
        
        top_n_index = torch.cat([top_n_index_small, top_n_index_large], 1)
        top_n_prob = torch.cat([top_n_prob_small, top_n_prob_large], 1)
        
        if return_vis:
            temp = temp.view(batch, self.topN, 2).data.cpu().numpy()
            temp = np.exp(temp)
            temp = temp / temp.sum(2, keepdims=True)
            temp = temp[:, :, 1]
            top_n_cdds = np.concatenate([top_n_cdds_small, top_n_cdds_large], 1)
            for i in range(batch):
                top_n_cdds[i, :, 0] = temp[i]
                    
            top_n_cdds = [hard_nms(x, topn=2, iou_thresh=0.1) for x in top_n_cdds]
            img_vis = vis(img_raw, top_n_cdds)
            try:
                anchor_lst = np.array(top_n_cdds)[:, :2]
            except:
                anchor_lst = np.array(top_n_cdds)[:, :2]
        # concat_logits have the shape: B*200
        concat_out = torch.cat([part_feature, feature, feature2], dim=1)
        concat_logits = self.concat_net(concat_out)
        raw_logits = resnet_out# (resnet_out + att_logits) / 2
        # part_logits have the shape: B*N*200
        part_logits = self.partcls_net(part_features).view(batch, self.topN, -1)
        if return_vis:
            return [raw_logits, concat_logits, part_logits, 
                    top_n_index, top_n_prob, img_vis, anchor_lst]
        else:
            return [raw_logits, concat_logits, part_logits, 
                top_n_index, top_n_prob]

def get_xy(y0, x0, y1, x1, size=448):
    pad_size = size//2
    
    y0 = np.max([y0, pad_size])
    y0 = np.min([y0, size+pad_size])
    
    x0 = np.max([x0, pad_size])
    x0 = np.min([x0, size+pad_size])
    
    y1 = np.max([y1, pad_size])
    y1 = np.min([y1, size+pad_size])
    
    x1 = np.max([x1, pad_size])
    x1 = np.min([x1, size+pad_size])
    
    return y0, x0, y1, x1


def list_loss(logits, targets):
    temp = F.log_softmax(logits, -1)
    loss = [-temp[i][targets[i].item()] for i in range(logits.size(0))]
    return torch.stack(loss)


def ranking_loss(score, targets, proposal_num=PROPOSAL_NUM):
    loss = Variable(torch.zeros(1).cuda())
    batch_size = score.size(0)
    for i in range(proposal_num):
        targets_p = (targets > targets[:, i].unsqueeze(1)).type(torch.cuda.FloatTensor)
        pivot = score[:, i].unsqueeze(1)
        loss_p = (1 - pivot + score) * targets_p
        loss_p = torch.sum(F.relu(loss_p))
        loss += loss_p
    return loss / batch_size


def vis(raw_img, top_n_cdds, n=2, T=0):
    raw_img = raw_img.data.cpu().numpy()
    img_lst = []
    for bs in range(raw_img.shape[0]):
        img = nor(raw_img[bs])
        for i in range(n):
            [y0, x0, y1, x1] = top_n_cdds[bs][i, 1:5].astype(np.int)
            y0, x0, y1, x1 = get_xy(y0, x0, y1, x1)
            y0 = int((y0-224)*600/448)
            x0 = int((x0-224)*600/448)
            y1 = int((y1-224)*600/448)
            x1 = int((x1-224)*600/448)
            img = draw(img, y0, x0, y1, x1)
        img_lst.append(img)
    return np.stack(img_lst)

def draw(img, y0, x0, y1, x1, w=8):
    R = 254
    G = 67
    B = 101
# =============================================================================
#     R = 1
#     G = 1
#     B = 1
# =============================================================================
    for i in range(w):
        if x0+i < img.shape[-1]:
            img[0, y0:y1, x0+i] = R
            img[1, y0:y1, x0+i] = G
            img[2, y0:y1, x0+i] = B
        
        if x1+i < img.shape[-1]:
            img[0, y0:y1, x1+i] = R
            img[1, y0:y1, x1+i] = G
            img[2, y0:y1, x1+i] = B
        
        if y0+i < img.shape[-1]:
            img[0, y0+i, x0:x1] = R
            img[1, y0+i, x0:x1] = G
            img[2, y0+i, x0:x1] = B
        
        if y1+i < img.shape[-1]:
            img[0, y1+i, x0:x1+w] = R
            img[1, y1+i, x0:x1+w] = G
            img[2, y1+i, x0:x1+w] = B
    
    return img

def nor(arr):
    result = arr.copy()
    for i in range(arr.shape[0]):
        MAX = arr[i].max()
        MIN = arr[i].min()
        result[i] = (arr[i] - MIN) / (MAX - MIN)
        result[i] = np.where(arr[i]==0, 0, result[i])
    result = result*255
    result = result.astype(np.uint8)
    return result