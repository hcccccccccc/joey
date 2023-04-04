import torch
from torch.utils.data import Dataset
from torch.utils import data
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
import numpy as np
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
        self.root = glob.glob(os.path.join(self.data_root, '*t1.nii'))
        self.t1 = sorted(glob.glob(os.path.join(self.data_root, '*_t1.nii')))
        self.t2 = sorted(glob.glob(os.path.join(self.data_root, '*_t1ce.nii')))
        self.t1ce = sorted(glob.glob(os.path.join(self.data_root, '*_t2.nii')))
        self.flair = sorted(glob.glob(os.path.join(self.data_root, '*_flair.nii')))
        self.labels = sorted(glob.glob(os.path.join(self.data_root, '*_seg.nii')))

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


class BratsDataset(Dataset):
    def __init__(self, data_dir):
        self.data_dir = data_dir
        self.subjects = os.listdir(data_dir)

        self.transform = BratsTransform()

    def __len__(self):
        return len(self.subjects)

    def __getitem__(self, idx):
        subject_dir = os.path.join(self.data_dir, self.subjects[idx])
        # print(subject_dir)
        # file = os.listdir(subject_dir)
        # print(len(file))
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
        image, label = self.transform((image, label))

        # Convert numpy arrays to PyTorch tensors
        image = torch.from_numpy(image).float()
        # label = torch.from_numpy(label).float()

        # return image, label
        return [image[0][0], image[1][0], image[2][0], image[3][0]], label


class BratsTransform:
    def __call__(self, sample):
        image, label = sample

        image[image < 0] = 0
        label = self.convert_seg_to_one_hot(label)
        label = torch.squeeze(label)
        # Crop image and label to 128x128x128
        image = image[:, :, 16:144, 56:184, 56:184]
        label = label[:, 16:144, 56:184, 56:184]
        
        # image = image.astype(np.float32)
        # label = label.astype(np.float32)

        # Randomly flip image and label
        # if np.random.rand() > 0.5:
        #     image = np.flip(image, axis=2)
        #     label = np.flip(label, axis=1)

        # if np.random.rand() > 0.5:
        #     image = np.flip(image, axis=3)
        #     label = np.flip(label, axis=2)

        # if np.random.rand() > 0.5:
        #     image = np.flip(image, axis=4)
        #     label = np.flip(label, axis=3)

        # Convert label to one-hot encoding    
        # label = torch.from_numpy(label).long()
        # # one_hot = torch.zeros(3, *label[0].shape)
        # # one_hot.scatter_(0, label, 1)
        # label = torch.nn.functional.one_hot(label, num_classes=3).permute(3, 0, 1, 2).float()


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
