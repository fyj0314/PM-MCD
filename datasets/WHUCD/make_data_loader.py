import os
import imageio
import argparse
import numpy as np

from torch.autograd import Variable
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torchvision import transforms

import datasets.WHUCD.imutils as imutils


augment_transforms = transforms.Compose([
    imutils.RandomHorizontalFlip(),
    imutils.RandomVerticalFlip(),
    imutils.RandomFixRotate()
])

def img_loader(path):
    img = np.array(imageio.v2.imread(path), np.float32)
    return img

class Datset(Dataset):
    def __init__(self, dataset_path, data_list, crop_size, max_iters=None, type='train', data_loader=img_loader):
        self.dataset_path = dataset_path
        self.data_list = data_list
        self.loader = data_loader
        self.type = type
        self.data_pro_type = self.type

        if max_iters is not None:
            self.data_list = self.data_list * int(np.ceil(float(max_iters) / len(self.data_list)))
            self.data_list = self.data_list[0:max_iters]
        self.crop_size = crop_size

    def __transforms(self, aug, pre_img, post_img, label):
        if aug:
            pre_img, post_img, label = augment_transforms([pre_img, post_img, label])

        pre_img = imutils.normalize_img(pre_img)  # imagenet normalization
        pre_img = np.transpose(pre_img, (2, 0, 1))

        post_img = imutils.normalize_img(post_img)  # imagenet normalization
        post_img = np.transpose(post_img, (2, 0, 1))

        return pre_img, post_img, label

    def __getitem__(self, index):
        if 'train' in self.data_pro_type:
            pre_path = os.path.join(self.dataset_path, 'A', self.data_list[index])
            post_path = os.path.join(self.dataset_path, 'B', self.data_list[index])
            label_path = os.path.join(self.dataset_path, 'label', self.data_list[index])
        else:
            pre_path = os.path.join(self.dataset_path, 'A', self.data_list[index])
            post_path = os.path.join(self.dataset_path, 'B', self.data_list[index])
            label_path = os.path.join(self.dataset_path, 'label', self.data_list[index])

        pre_img = self.loader(pre_path)
        post_img = self.loader(post_path)
        label = self.loader(label_path)
        label = label / 255

        if 'train' in self.data_pro_type:
            pre_img, post_img, label = self.__transforms(True, pre_img, post_img, label)
        else:
            pre_img, post_img, label = self.__transforms(False, pre_img, post_img, label)
            label = np.asarray(label)

        data_idx = self.data_list[index]
        return pre_img, post_img, label, data_idx

    def __len__(self):
        return len(self.data_list)


def make_data_loader(args, **kwargs):  # **kwargs could be omitted
    dataset = Datset(args.train_dataset_path, args.train_data_name_list, args.crop_size, args.max_iters, args.type)
    data_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=args.shuffle, **kwargs, num_workers=16,
                             drop_last=False)
    return data_loader


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="Landsat DataLoader Test")
    parser.add_argument('--dataset', type=str, default='CNAM-CD')
    parser.add_argument('--max_iters', type=int, default=3000)
    parser.add_argument('--type', type=str, default='train')
    parser.add_argument('--train_dataset_path', type=str, default='D:\DataSets\Landsat\\train')
    parser.add_argument('--train_data_list_path', type=str, default='D:\PycharmProjects\GCFN3\datasets\LandsatSCD\\train_list.txt')
    parser.add_argument('--shuffle', type=bool, default=True)
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--train_data_name_list', type=list)
    parser.add_argument('--crop_size', type=int, default=512)

    args = parser.parse_args()

    with open(args.train_data_list_path, "r") as f:
        # data_name_list = f.read()
        train_data_name_list = [train_data_name.strip() for train_data_name in f]
    args.train_data_name_list = train_data_name_list

    train_data_loader = make_data_loader(args)
    for i, data in enumerate(train_data_loader):
        pre_img, post_img, labels, _ = data
        pre_data, post_data = Variable(pre_img), Variable(post_img)
        labels = Variable(labels)
        print(i, "个inputs", pre_data.data.size(), "labels", labels.data.size())
