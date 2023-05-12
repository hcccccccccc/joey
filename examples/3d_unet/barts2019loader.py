import torch
from torch.utils.data import Dataset
from torch.utils import data
import torchvision.transforms.functional as TF
import numpy as np
import os
import glob
import SimpleITK as sitk


class BratsDataset(Dataset):
    def __init__(self, data_dir, data_size, model_version):
        self.data_dir = data_dir
        self.subjects = os.listdir(data_dir)
        self.data_size = data_size
        self.model_version = model_version
        self.transform = BratsTransform()

    def __len__(self):
        return len(self.subjects)

    def __getitem__(self, idx):
        subject_dir = os.path.join(self.data_dir, self.subjects[idx])

        f_t1 = glob.glob(os.path.join(subject_dir, '*_t1.nii'))
        f_t1ce = glob.glob(os.path.join(subject_dir, '*_t1ce.nii'))
        f_t2 = glob.glob(os.path.join(subject_dir, '*_t2.nii'))
        f_flair = glob.glob(os.path.join(subject_dir, '*_flair.nii'))
        f_seg = glob.glob(os.path.join(subject_dir, '*_seg.nii'))

        # Load nifti files for each modality
        t1_img = sitk.ReadImage(f_t1)
        t1ce_img = sitk.ReadImage(f_t1ce)
        t2_img = sitk.ReadImage(f_t2)
        flair_img = sitk.ReadImage(f_flair)
        t1_img = sitk.GetArrayFromImage(t1_img)
        t1ce_img = sitk.GetArrayFromImage(t1ce_img)
        t2_img = sitk.GetArrayFromImage(t2_img)
        flair_img = sitk.GetArrayFromImage(flair_img)
        # Stack modalities into 4D tensor
        image = np.stack([t1_img, t1ce_img, t2_img, flair_img], axis=0)

        # Normalize image
        image = (image - image.mean()) / image.std()

        # Load segmentation mask
        seg_img = sitk.ReadImage(f_seg)
        label = sitk.GetArrayFromImage(seg_img)

        # Apply data augmentation
        image, label = self.transform((image, label), self.data_size, self.model_version)

        # Convert numpy arrays to PyTorch tensors
        image = torch.from_numpy(image).float()
        # label = torch.from_numpy(label).float()

        # return image, label
        return [image[0][0], image[1][0], image[2][0], image[3][0]], label


class BratsTransform:
    def __call__(self, sample, size, model):
        image, label = sample

        image[image < 0] = 0
        label = self.convert_seg_to_one_hot(label)
        label = torch.squeeze(label)
        start = []
        start.append(int((label.shape[1]-size)/2))
        start.append(int((label.shape[2]-size)/2))
        if model == 'joey':
            image = image[:,:,start[0]:start[0]+size,start[1]:start[1]+size,start[1]:start[1]+size]
            label = label[:,start[0]:start[0]+size+2,start[1]:start[1]+size+2,start[1]:start[1]+size+2]
        else:
            image = image[:,:,start[0]:start[0]+size,start[1]:start[1]+size,start[1]:start[1]+size]
            label = label[:,start[0]:start[0]+size,start[1]:start[1]+size,start[1]:start[1]+size]

        return image, label
    
    def convert_seg_to_one_hot(self, seg):
        et = (seg == 4).astype(int)
        tc = ((seg == 1) | (seg == 4)).astype(int) - et
        wt = (seg > 0).astype(int) - et - tc
        seg_one_hot = np.stack((et, tc, wt), axis=0)
        return torch.tensor(seg_one_hot)

if __name__=='__main__':
    data_root = '/run/datasets/MICCAI_BraTS_2019_Data_Training'
    BraTS2019 = BratsDataset(data_root)
    train_loader = data.DataLoader(BraTS2019, batch_size=4, shuffle=False, num_workers=4)
    for image, label in train_loader:
        pil_img = TF.to_pil_image(label[0][1][64].float())
        pil_img.save("my_image.jpg", format="JPEG")
