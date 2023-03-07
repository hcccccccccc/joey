import torch
from torch.utils.data import Dataset
from torch.utils import data
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import itk
import os
import glob
import SimpleITK as sitk

class dataset(Dataset):
    def __init__(self, data_root):
        super().__init__()
        self.data_root = data_root
        self.list_path_data = os.listdir(data_root)
        train_transformer = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize(256),
            # transforms.RandomResizedCrop(224,scale=(0.5,1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        root = glob.glob(os.path.join(self.data_root, '*/*t1.nii'))
        self.t1 = sorted(glob.glob(os.path.join(self.data_root, '*/*_t1.nii')))
        self.t2 = sorted(glob.glob(os.path.join(self.data_root, '*/*_t1ce.nii')))
        self.t1ce = sorted(glob.glob(os.path.join(self.data_root, '*/*_t2.nii')))
        self.flair = sorted(glob.glob(os.path.join(self.data_root, '*/*_flair.nii')))
        self.labels = sorted(glob.glob(os.path.join(self.data_root, '*/*_seg.nii')))

    def __getitem__(self, index):
        f_t1 = sitk.ReadImage(self.t1[index])
        f_t1ce = sitk.ReadImage(self.t1ce[index])
        f_t2 = sitk.ReadImage(self.t2[index])
        f_flair = sitk.ReadImage(self.flair[index])
        f_labels = sitk.ReadImage(self.labels[index])
        img_t1 = sitk.GetArrayFromImage(f_t1)
        img_t1ce = sitk.GetArrayFromImage(f_t1ce)
        img_t2 = sitk.GetArrayFromImage(f_t2)
        img_flair = sitk.GetArrayFromImage(f_flair)
        img_labels = sitk.GetArrayFromImage(f_labels)

        return [img_t1, img_t1ce, img_t2, img_flair], img_labels

    def __len__(self):
        return len(self.list_path_data)


# if __name__=='__main__':
    # data_root = './joey/examples/3d_unet/data'
    # BraTS2019 = dataset(data_root)
    # train_loader = data.DataLoader(BraTS2019, batch_size=2, shuffle=True, num_workers=4,
    #                                      pin_memory=False)
    # for t1, t1ce, t2, flair, labels in train_loader:
    #     print(t1.shape)
