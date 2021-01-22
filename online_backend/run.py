# -*- coding: utf-8 -*-
"""
Created on Tue Feb 11 14:35:24 2020

@author: SY
"""
import sys
import torch
import os
use_gpu = True
if torch.cuda.is_available():
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
else:
    # assert 0, 'Please check gpu is available!'
    use_gpu = False
import numpy as np
from utils.dataset import SARS
from utils.model import attention_net as net
from utils.model import get_xy, draw
from utils.pre_cut_lungs import get_segmented_lungs
from PIL import Image

def run_preprocess(path):
    vis_img_lst = []
    test_lst = []
    img_fp = []
    for fn in os.listdir(path):
        fp = os.path.join(path, fn)
        flg, test_img, train_img = get_segmented_lungs(fp)
        if not flg:
            continue
        vis_img_lst.append(train_img)
        test_lst.append(test_img)
        img_fp.append(fp)
    return test_lst, vis_img_lst, img_fp


def run_model(img_lst, vis_img_lst, img_fp):
    model = net(topN=6, n_class=2, use_gpu=use_gpu)
    if use_gpu:
        model.load_state_dict(torch.load('./model.ckpt')['net_state_dict'])
        model = model.cuda()
    else:
        model.load_state_dict(torch.load('./model.ckpt', map_location='cpu')['net_state_dict'])
    model.eval()
    loader = load_data(img_lst)
    
    pred_lst = []
    anchor_lst = []
    for i, data in enumerate(loader):
        if use_gpu:
            img, img_raw = data[0].cuda(), data[2].cuda()
        else:
            img, img_raw = data[0], data[2]
        with torch.no_grad():
            _, concat_logits, _, _, _, anchor = model(img, img_raw, True, True)
        # calculate accuracy
        pred = torch.nn.Softmax(1)(concat_logits)
        pred_lst.append(pred.data.cpu().numpy())
        anchor_lst.append(anchor)
    pred_lst = np.concatenate(pred_lst, 0)
    anchor_lst = np.concatenate(anchor_lst, 0)
    
    T = 3
    topk = 6
    score = anchor_lst[:, 0, 0].reshape(-1)
    rank = [index for index, value in 
            sorted(list(enumerate(score)), key=lambda x: x[1], reverse=True)]
    rank = rank[:topk]
    counter_save = 0
    for i in rank:
        fp = img_fp[i]
        img = vis_img_lst[i].transpose(2,0,1)
        y, x = img.shape[1], img.shape[2]
        flg = 0
        for j in range(anchor_lst.shape[1]):
            if anchor_lst[i][j][0] < T:
                continue
            flg = 1
            [y0, x0, y1, x1] = anchor_lst[i, j, 1:5].astype(np.int)
            y0, x0, y1, x1 = get_xy(y0, x0, y1, x1)
            y0 = int((y0-224)*y/448)
            x0 = int((x0-224)*x/448)
            y1 = int((y1-224)*y/448)
            x1 = int((x1-224)*x/448)
            img = draw(img, y0, x0, y1, x1)
        if flg == 0:
            continue
        img = img.transpose(1,2,0)
        img = Image.fromarray(img, 'RGB')
        end = '.'+fp.split('.')[-1]
        save_fp = fp.replace(end, '_vis.jpg')
        img.save(save_fp)
        counter_save += 1
    while counter_save == 0:
        T -= 0.3
        counter_save = 0
        for i in rank:
            fp = img_fp[i]
            img = vis_img_lst[i].transpose(2,0,1)
            y, x = img.shape[1], img.shape[2]
            flg = 0
            for j in range(anchor_lst.shape[1]):
                if anchor_lst[i][j][0] < T:
                    continue
                flg = 1
                [y0, x0, y1, x1] = anchor_lst[i, j, 1:5].astype(np.int)
                y0, x0, y1, x1 = get_xy(y0, x0, y1, x1)
                y0 = int((y0-224)*y/448)
                x0 = int((x0-224)*x/448)
                y1 = int((y1-224)*y/448)
                x1 = int((x1-224)*x/448)
                img = draw(img, y0, x0, y1, x1)
            if flg == 0:
                continue
            img = img.transpose(1,2,0)
            img = Image.fromarray(img, 'RGB')
            end = '.'+fp.split('.')[-1]
            save_fp = fp.replace(end, '_vis.jpg')
            img.save(save_fp)
            counter_save += 1
    return list(pred_lst[:, 1])
        

def run_people_level_prediction(pred_lst):
    return np.mean(pred_lst)


def load_data(img_lst, BATCH_SIZE=4):
    dataset = SARS(img_lst=img_lst, is_train=False)
    loader = torch.utils.data.DataLoader(
            dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=1, drop_last=False)
    return loader


def main(path):
    img_lst, vis_img_lst, img_fp = run_preprocess(path)
    pred_lst = run_model(img_lst, vis_img_lst, img_fp)
    result = run_people_level_prediction(pred_lst)
    return result


if __name__ == '__main__':
    path = sys.argv[-1]
    # print(path)
    result = main(path)
    # result = '{prediction:'+f'{result:.2f}'+'}'
    print(result)