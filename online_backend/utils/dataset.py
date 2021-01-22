import numpy as np
from PIL import Image
from torchvision import transforms
INPUT_SIZE = (448, 448)
import random
random.seed(0)

def clean_name(name):
    return name

class SARS():
    def __init__(self, img_lst, is_train=True):
        self.is_train = is_train
        self.img_lst = img_lst
        
        
        
    def __getitem__(self, index):
        flg_H = 0
        img = np.array(self.img_lst[index])
        target = -1
        if len(img.shape) == 2:
            img = np.stack([img] * 3, 2)
        img_raw = img.copy()
        img = Image.fromarray(img, mode='RGB')
        img = transforms.Resize((448, 448), Image.BILINEAR)(img)
# =============================================================================
#         if self.is_train:
#             img = transforms.RandomCrop(INPUT_SIZE)(img)
#             if np.random.randint(2) == 1:
#                 flg_H = 1
#                 img = transforms.RandomHorizontalFlip(p=1)(img)
#             img = transforms.ColorJitter(brightness=0.126, saturation=0.5)(img)
#         else:
#             img = transforms.CenterCrop(INPUT_SIZE)(img)
#             pass
# =============================================================================
        img = transforms.ToTensor()(img)
        img = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])(img)
        img_raw = Image.fromarray(img_raw, mode='RGB')
        img_raw = transforms.Resize((600, 600), Image.BILINEAR)(img_raw)
        if flg_H == 1:
            img_raw = transforms.RandomHorizontalFlip(p=1)(img_raw)
        
        img_raw = transforms.ToTensor()(img_raw)
        img_raw = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])(img_raw)
        return img, target, img_raw

    def __len__(self):
        return len(self.img_lst)

