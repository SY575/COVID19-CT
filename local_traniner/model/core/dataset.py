import numpy as np
import scipy.misc
import os
from PIL import Image
from torchvision import transforms
from config import INPUT_SIZE
from tqdm import tqdm
from scipy.io import loadmat
import random
random.seed(0)

def clean_name(name):
    return name
    name = name.replace(' ', '').replace('.jpg', '')
    try:
        int(name)
        return name
    except:
        pass
    while True:
        try:
            int(name[-1])
            name = name[:-1]
        except:
            return name

class SARS():
    def __init__(self, root, is_train=True):
        self.is_train = is_train
        
        path = f'{root}/no_nCoV'
        self.file_lst = [f'{path}/{item}' for item in os.listdir(path)]
        self.people_lst = [clean_name(item.split('_')[0])
                            for item in os.listdir(path)]
        self.label_lst = [0]*len(os.listdir(path))
        
        path = f'{root}/nCoV'
        self.file_lst += [f'{path}/{item}' for item in os.listdir(path)]
        self.people_lst += [clean_name(item.split('_')[0])
                            for item in os.listdir(path)]
        self.label_lst += [1]*len(os.listdir(path))
        
        
        temp = list(zip(self.file_lst, self.label_lst, self.people_lst))
        random.shuffle(temp)
        self.file_lst = [item[0] for item in temp]
        self.label_lst = [item[1] for item in temp]
        self.people_lst = [item[2] for item in temp]
        self.file_name = [item.split('/')[-1] for item in self.file_lst]
        
    def __getitem__(self, index):
        flg_H = 0
        img = np.array(Image.open(self.file_lst[index]))
        target = self.label_lst[index]
        if len(img.shape) == 2:
            img = np.stack([img] * 3, 2)
        img_raw = img.copy()
        img = Image.fromarray(img, mode='RGB')
        img = transforms.Resize((448, 448), Image.BILINEAR)(img)
        if self.is_train:
# =============================================================================
#             img = transforms.RandomCrop(INPUT_SIZE)(img)
# =============================================================================
            if np.random.randint(2) == 1:
                flg_H = 1
                img = transforms.RandomHorizontalFlip(p=1)(img)
            img = transforms.ColorJitter(brightness=0.126, saturation=0.5)(img)
        else:
# =============================================================================
#             img = transforms.CenterCrop(INPUT_SIZE)(img)
# =============================================================================
            pass
        img = transforms.ToTensor()(img)
        img = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])(img)
        img_raw = Image.fromarray(img_raw, mode='RGB')
        img_raw = transforms.Resize((600, 600), Image.BILINEAR)(img_raw)
        if flg_H == 1:
            img_raw = transforms.RandomHorizontalFlip(p=1)(img_raw)
        
        img_raw = transforms.ToTensor()(img_raw)
        img_raw = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])(img_raw)
        return img, target, img_raw, self.people_lst[index], self.file_name[index]

    def __len__(self):
        return len(self.label_lst)

class CUB():
    def __init__(self, root, is_train=True, data_len=None):
        self.root = root
        self.is_train = is_train
        img_txt_file = open(os.path.join(self.root, 'images.txt'))
        label_txt_file = open(os.path.join(self.root, 'image_class_labels.txt'))
        train_val_file = open(os.path.join(self.root, 'train_test_split.txt'))
        img_name_list = []
        for line in img_txt_file:
            img_name_list.append(line[:-1].split(' ')[-1])
        label_list = []
        for line in label_txt_file:
            label_list.append(int(line[:-1].split(' ')[-1]) - 1)
        train_test_list = []
        for line in train_val_file:
            train_test_list.append(int(line[:-1].split(' ')[-1]))
        self.train_file_list = [x for i, x in zip(train_test_list, img_name_list) if i]
        self.test_file_list = [x for i, x in zip(train_test_list, img_name_list) if not i]
        if self.is_train:
            self.train_label = [x for i, x in zip(train_test_list, label_list) if i][:data_len]
        if not self.is_train:
            self.test_label = [x for i, x in zip(train_test_list, label_list) if not i][:data_len]

    def __getitem__(self, index):
        flg_H = 0
# =============================================================================
#         flg_V = 0
# =============================================================================
        
        if self.is_train:
            img = np.array(Image.open(os.path.join(self.root, 'images', self.train_file_list[index])))
            target = self.train_label[index]
# =============================================================================
#             img, target = self.train_img[index], self.train_label[index]
# =============================================================================
            if len(img.shape) == 2:
                img = np.stack([img] * 3, 2)
            img_raw = img.copy()
            img = Image.fromarray(img, mode='RGB')
            img = transforms.Resize((600, 600), Image.BILINEAR)(img)
            img = transforms.RandomCrop(INPUT_SIZE)(img)
# =============================================================================
#             img = transforms.CenterCrop(INPUT_SIZE)(img)
# =============================================================================
            if np.random.randint(2) == 1:
                flg_H = 1
                img = transforms.RandomHorizontalFlip(p=1)(img)
# =============================================================================
#             if np.random.randint(2) == 1:
#                 flg_V = 1
#                 img = transforms.RandomVerticalFlip(p=1)(img)
# =============================================================================
            img = transforms.ColorJitter(brightness=0.126, saturation=0.5)(img)
# =============================================================================
#             img = transforms.ColorJitter(brightness=(0, 1), 
#                                          contrast=(0, 1), 
#                                          saturation=(0, 1))(img)
# =============================================================================
            img = transforms.ToTensor()(img)
            img = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])(img)

        else:
            img = np.array(Image.open(os.path.join(self.root, 'images', self.test_file_list[index])))
            target = self.test_label[index]
# =============================================================================
#             img, target = self.test_img[index], self.test_label[index]
# =============================================================================
            if len(img.shape) == 2:
                img = np.stack([img] * 3, 2)
            img_raw = img.copy()
            img = Image.fromarray(img, mode='RGB')
            img = transforms.Resize((600, 600), Image.BILINEAR)(img)
            img = transforms.CenterCrop(INPUT_SIZE)(img)
            img = transforms.ToTensor()(img)
            img = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])(img)
        img_raw = Image.fromarray(img_raw, mode='RGB')
        img_raw = transforms.Resize((600, 600), Image.BILINEAR)(img_raw)
        if flg_H == 1:
            img_raw = transforms.RandomHorizontalFlip(p=1)(img_raw)
# =============================================================================
#         if flg_V == 1:
#             img_raw = transforms.RandomVerticalFlip(p=1)(img_raw)
# =============================================================================
        
        img_raw = transforms.ToTensor()(img_raw)
        img_raw = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])(img_raw)
        return img, target, img_raw

    def __len__(self):
        if self.is_train:
            return len(self.train_label)
        else:
            return len(self.test_label)

class Car():
    def __init__(self, root, is_train=True, data_len=None):
        self.DATAPATH = root
        if is_train:
            self.phase = 'train'
        else:
            self.phase = 'test'
        self.resize = (448, 448)
        self.num_classes = 196
        
        if self.phase == 'train':
            list_path = os.path.join(self.DATAPATH, 'devkit', 'cars_train_annos.mat')
            self.image_path = os.path.join(self.DATAPATH, 'cars_train')
        else:
            list_path = os.path.join(self.DATAPATH, 'cars_test_annos_withlabels.mat')
            self.image_path = os.path.join(self.DATAPATH, 'cars_test')

        list_mat = loadmat(list_path)
        self.images = [f.item() for f in list_mat['annotations']['fname'][0]]
        self.labels = [f.item() for f in list_mat['annotations']['class'][0]]
        

    def __getitem__(self, item):
        # image
        img = Image.open(os.path.join(self.image_path, self.images[item])).convert('RGB')  # (C, H, W)
        img_raw = img.copy()
        flg_H = 0
        flg_V = 0
        img = transforms.Resize(size=(int(self.resize[0] / 0.875), 
                                      int(self.resize[1] / 0.875)))(img) 
        if self.phase == 'train':
            img = transforms.RandomCrop(self.resize)(img)
            if np.random.randint(2) == 1:
                flg_H = 1
                img = transforms.RandomHorizontalFlip(p=1)(img)
            if np.random.randint(2) == 1:
                flg_V = 1
                img = transforms.RandomVerticalFlip(p=1)(img)
            img = transforms.ColorJitter(brightness=0.126, saturation=0.5)(img)
        else:
            img = transforms.CenterCrop(self.resize)(img)
        img = transforms.ToTensor()(img)
        img = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])(img)
# =============================================================================
#         
# =============================================================================
        img_raw = transforms.Resize((600, 600), Image.BILINEAR)(img_raw)
        if flg_H == 1:
            img_raw = transforms.RandomHorizontalFlip(p=1)(img_raw)
        if flg_V == 1:
            img_raw = transforms.RandomVerticalFlip(p=1)(img_raw)
        img_raw = transforms.ColorJitter(brightness=0.126, saturation=0.5)(img_raw)
        img_raw = transforms.ToTensor()(img_raw)
        img_raw = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])(img_raw)
        # return image and label
        return img, self.labels[item] - 1, img_raw

    def __len__(self):
        return len(self.images)

if __name__ == '__main__':
    dataset = CUB(root='./CUB_200_2011')
    print(len(dataset.train_img))
    print(len(dataset.train_label))
    for data in dataset:
        print(data[0].size(), data[1])
    dataset = CUB(root='./CUB_200_2011', is_train=False)
    print(len(dataset.test_img))
    print(len(dataset.test_label))
    for data in dataset:
        print(data[0].size(), data[1])
