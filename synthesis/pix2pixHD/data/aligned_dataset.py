import os
from data.base_dataset import BaseDataset, get_params, get_transform
from data.image_folder import make_dataset
from PIL import Image
import numpy as np
import torch


class AlignedDataset(BaseDataset):
    def initialize(self, opt):
        self.opt = opt
        self.root = opt.dataroot

        dir_A = '_A' if self.opt.phase == 'train' else '_A'
        dir_B = '_B' if self.opt.phase == 'train' else '_B'

        self.dir_A = os.path.join(self.root, opt.phase + dir_A)
        self.dir_B = os.path.join(self.root, opt.phase + dir_B)

        self.A_paths = sorted(make_dataset(self.dir_A))
        self.B_paths = sorted(make_dataset(self.dir_B))

        self.A_size = len(self.A_paths)
        self.B_size = len(self.B_paths)

    def __getitem__(self, index):
        # 读取 A（p0 RGB）
        A_path = self.A_paths[index % self.A_size]
        A_img = Image.open(A_path).convert('RGB')
        params = get_params(self.opt, A_img.size)
        transform_A = get_transform(self.opt, params)
        A_tensor = transform_A(A_img)

        # 读取 B（p1~p5 npy 或 RGB）
        B_path = self.B_paths[index % self.B_size]
        if B_path.lower().endswith('.npy'):
            arr = np.load(B_path)  # HxWx5
            chans = []
            transform_B = get_transform(self.opt, params, grayscale=True)
            for c in range(arr.shape[-1]):
                im = Image.fromarray(arr[..., c].astype(np.uint8), mode='L')
                chans.append(transform_B(im))
            B_tensor = torch.cat(chans, dim=0)  # 5xHxW
        else:
            B_img = Image.open(B_path).convert('RGB')
            transform_B = get_transform(self.opt, params)
            B_tensor = transform_B(B_img)

        return {
            'label': A_tensor,  # p0 输入
            'inst': torch.zeros(1, A_tensor.size(1), A_tensor.size(2)),  # 占位
            'image': B_tensor,  # p1~p5 输出
            'feat': torch.zeros(1),  # 假的 feature，占位用
            'path': A_path
        }


    def __len__(self):
        return max(self.A_size, self.B_size)

    def name(self):
        return 'AlignedDataset'
