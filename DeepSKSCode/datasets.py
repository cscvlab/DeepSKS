import torch
from torch.utils.data import Dataset, DataLoader
import os, json, cv2
import numpy as np
from glob import glob
import matplotlib.pyplot as plt
from homo_utils import generate_homo, cal_Hs2_Hs2_inv


class MSCOCO(Dataset):
    def __init__(self, split):
        self.homo_parameter = {"marginal": 32, "perturb": 32, "patch_size": 128}

        if split == 'train':
            root_img2 = './DataSet/MSCOCO2017/train2017'
            root_img1 = './DataSet/MSCOCO2017/train2017'
        else:
            root_img2 = './DataSet/MSCOCO2017/test2017'
            root_img1 = './DataSet/MSCOCO2017/test2017'

        self.image_list_img1 = sorted(glob(os.path.join(root_img1, '*.jpg')))
        self.image_list_img2 = sorted(glob(os.path.join(root_img2, '*.jpg')))

    def __len__(self):
        return len(self.image_list_img1)

    def __getitem__(self, index):
        img1 = cv2.imread(self.image_list_img1[index])
        img2 = cv2.imread(self.image_list_img2[index])
        img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
        img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)

        img1 = cv2.resize(img1, (320, 240))
        img2 = cv2.resize(img2, (320, 240))

        self.homo_parameter["height"], self.homo_parameter["width"], _ = img1.shape

        patch_img1_warp, patch_img2, four_gt, org_pts, dst_pts, param_gt, angle_gt, large_img1_warp, large_img2 = generate_homo(
            img1, img2, homo_parameter=self.homo_parameter, transform=None)

        return {"patch_img1_warp": patch_img1_warp, "patch_img2": patch_img2, "four_gt": four_gt,
                "org_pts": org_pts, "dst_pts": dst_pts, "param_gt": param_gt, "angle_gt": angle_gt,
                "large_img1_warp": large_img1_warp, "large_img2": large_img2}


class homo_dataset(Dataset):
    def __init__(self, split, dataset, args):
        self.dataset = dataset
        self.args = args
        self.homo_parameter = {"marginal": 32, "perturb": 32, "patch_size": 128}

        if split == 'train':
            if dataset == 'GoogleMap':
                root_img1 = './DataSet/GoogleMap/train2014_input'
                root_img2 = './DataSet/GoogleMap/train2014_template_original'
            if dataset == 'SPID':
                root_img1 = './DataSet/SPID/img_pair_train_new/img1'
                root_img2 = './DataSet/SPID/img_pair_train_new/img2'

        else:
            if dataset == 'GoogleMap':
                root_img1 = './DataSet/GoogleMap/val2014_input'
                root_img2 = './DataSet/GoogleMap/val2014_template_original'
            if dataset == 'SPID':
                root_img1 = './DataSet/SPID/img_pair_test_new/img1'
                root_img2 = './DataSet/SPID/img_pair_test_new/img2'

        self.image_list_img1 = sorted(glob(os.path.join(root_img1, '*.jpg')))
        self.image_list_img2 = sorted(glob(os.path.join(root_img2, '*.jpg')))

    def __len__(self):
        return len(self.image_list_img1)

    def __getitem__(self, index):
        img1 = cv2.imread(self.image_list_img1[index])
        img2 = cv2.imread(self.image_list_img2[index])

        img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
        img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)

        img_size = self.homo_parameter["patch_size"] + 2 * self.homo_parameter["marginal"]
        img1 = cv2.resize(img1, (img_size, img_size))
        img2 = cv2.resize(img2, (img_size, img_size))

        self.homo_parameter["height"], self.homo_parameter["width"], _ = img1.shape

        patch_img1_warp, patch_img2, four_gt, org_pts, dst_pts, param_gt, angle_gt, large_img1_warp, large_img2 = generate_homo(
            img1, img2, homo_parameter=self.homo_parameter, transform=None)

        return {"patch_img1_warp": patch_img1_warp, "patch_img2": patch_img2, "four_gt": four_gt,
                "org_pts": org_pts, "dst_pts": dst_pts, "param_gt": param_gt, "angle_gt": angle_gt,
                "large_img1_warp": large_img1_warp, "large_img2": large_img2}


def fetch_dataloader(args, split='test'):
    if args.dataset == "GoogleEarth":
        dataset = GoogleEarth(split=split)
    elif args.dataset == "MSCOCO":
        dataset = MSCOCO(split=split)
    else:
        dataset = homo_dataset(split=split, dataset=args.dataset, args=args)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, pin_memory=True, shuffle=True, num_workers=8,
                            drop_last=False)
    print('Test with %d image pairs' % len(dataset))

    return dataloader
