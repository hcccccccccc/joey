import torch
from torch.utils.data import Dataset
from torch.utils import data
import torchvision.transforms as transforms
import matplotlib.image as plt
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
        self.train_transformer = transforms.Compose([
            transforms.Resize((128,128)),
            transforms.ToTensor(),
        ])
        self.root = glob.glob(os.path.join(self.data_root, '*/*t1.nii'))
        self.t1 = sorted(glob.glob(os.path.join(self.data_root, '*/*_t1.nii')))
        self.t2 = sorted(glob.glob(os.path.join(self.data_root, '*/*_t1ce.nii')))
        self.t1ce = sorted(glob.glob(os.path.join(self.data_root, '*/*_t2.nii')))
        self.flair = sorted(glob.glob(os.path.join(self.data_root, '*/*_flair.nii')))
        self.labels = sorted(glob.glob(os.path.join(self.data_root, '*/*_seg.nii')))

    def __getitem__(self, index):
        f_t1 = self.resize(self.t1[index], (128,128,128))
        f_t1ce = self.resize(self.t1ce[index], (128,128,128))
        f_t2 = self.resize(self.t2[index], (128,128,128))
        f_flair = self.resize(self.flair[index], (128,128,128))
        f_labels = self.resize(self.labels[index], (128,128,128))
        img_t1 = sitk.GetArrayFromImage(f_t1)
        img_t1 = sitk.GetArrayFromImage(f_t1)
        img_t1ce = sitk.GetArrayFromImage(f_t1ce)
        img_t2 = sitk.GetArrayFromImage(f_t2)
        img_flair = sitk.GetArrayFromImage(f_flair)
        img_labels = sitk.GetArrayFromImage(f_labels)
        # print(img_labels[10])
        return [img_t1, img_t1ce, img_t2, img_flair], img_labels

    def __len__(self):
        return len(self.list_path_data)
    
    def resize(self, img, newsize, resamplemethod=sitk.sitkLinear):
            image = sitk.ReadImage(img)
            resampler=sitk.ResampleImageFilter()
            originsize = image.GetSize()
            originspacing = image.GetSpacing()
            newsize = np.array(newsize, float)
            factor = originsize / newsize
            newspacing = originspacing*factor
            newsize = newsize.astype(np.int)
            resampler.SetReferenceImage(image)
            resampler.SetSize(newsize.tolist())
            resampler.SetOutputSpacing(newspacing.tolist())
            resampler.SetTransform(sitk.Transform(3, sitk.sitkIdentity))
            resampler.SetInterpolator(resamplemethod)
            output = resampler.Execute(image)
            return output


if __name__=='__main__':
    data_root = './joey/examples/3d_unet/data'
    BraTS2019 = dataset(data_root)
    train_loader = data.DataLoader(BraTS2019, batch_size=2, shuffle=True, num_workers=4,
                                         pin_memory=False)
    for img, labels in train_loader:
        img = np.asarray(img)
        print(img.shape)
