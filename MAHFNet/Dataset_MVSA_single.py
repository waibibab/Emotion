

import torch
import numpy as np
from torch.utils.data import Dataset
from torch.utils import data
import re

from torchvision.transforms import InterpolationMode
BICUBIC = InterpolationMode.BICUBIC
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize, RandomVerticalFlip
from PIL import Image

def _convert_image_to_rgb(image):
    return image.convert("RGB")
def _transform(n_px):
    return Compose([
        Resize(n_px, interpolation=BICUBIC),
        CenterCrop(n_px),
        _convert_image_to_rgb,
        ToTensor(),
        Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
    ])

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
batch_size = 16


def dp_txt(txt):
    # nonEnglish_regex = re.compile('[^a-zA-Z0-9\\?\\!\\,\\.@#\\+\\-=\\*\'\"><&\\$%\\(\\)\\[\\]:;]+')
    hashtag_pattern = re.compile('#[a-zA-Z0-9]+')
    at_pattern = re.compile('@[a-zA-Z0-9]+')
    http_pattern = re.compile(
        "((http|ftp|https)://)(([a-zA-Z0-9\._-]+\.[a-zA-Z]{2,6})|([0-9]{1,3}\.[0-9]{1,3}\.[0-9]{1,3}\.[0-9]{1,3}))(:[0-9]{1,4})*(/[a-zA-Z0-9\&%_\./-~-]*)?")
    txt = txt.strip()
    txt_hashtag = re.sub(hashtag_pattern, '', txt)
    txt_nonat = re.sub(at_pattern, '', txt_hashtag)
    txt_nonhttp = re.sub(http_pattern, '', txt_nonat)
    txt = txt_nonhttp
    return txt


class MVSA_Single(Dataset):
    def __init__(self, txt_path, dp=False):
        with open(txt_path, 'r', encoding='utf-8') as fh:
            self.imgs = []
            for line in fh:
                # 使用制表符分割四列数据
                parts = line.strip().split('\t')
                
                # 跳过格式不正确的行（至少需要4列）
                if len(parts) < 4:
                    print(f"跳过格式错误行: {line.strip()}")
                    continue
                
                name, label, text, translation = parts[0], parts[1], parts[2], parts[3]
                
                # 转换标签为整数
                try:
                    label = int(label)
                except ValueError:
                    print(f"无效标签: {label}，行内容: {line.strip()}")
                    continue
                
                # 保留原始文本结构（包括空格）
                text = text.strip()
                translation = translation.strip()
                
                self.imgs.append((name, label, text, translation))
        
        self.dp = dp  # 保持原有的数据增强标志

    def __getitem__(self, index):
        name_path, emo_label, text, translation = self.imgs[index]
        if self.dp:
            text = dp_txt(text)
            translation = dp_txt(translation)
        image = _transform(n_px=224)(Image.open(name_path))
        return image, text, translation, emo_label

    def __len__(self):
        return len(self.imgs)


train_path = './MVSA_Single/train_0.9_des.txt'  #
test_path = './MVSA_Single/test_0.1_des.txt'  #
valid_path = './MVSA_Single/valid_0.1_des.txt'
train_d = MVSA_Single(train_path)
valid_d = MVSA_Single(valid_path)
test_d = MVSA_Single(test_path)

def get_dataset(batch_size = 16, drop_last=False):
    train_data = data.DataLoader(train_d, batch_size=batch_size, shuffle=False, drop_last=drop_last,num_workers=10)
    valid_data = data.DataLoader(valid_d, batch_size=batch_size, shuffle=False,num_workers=10)
    test_data = data.DataLoader(test_d, batch_size=batch_size, shuffle=False,num_workers=10)
    return train_data,valid_data, test_data

if __name__ == '__main__':
    # training
    train_data, test_data = get_dataset(64)
    for t in range(1):
        for i, data in enumerate(train_data):
            image, text, translation,label = data
            print(image.shape)

