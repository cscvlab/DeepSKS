import random
import numpy as np
import cv2
import torch
import torch.nn.functional as F
import torchgeometry as tgm


def generate_homo(img1, img2, homo_parameter, transform=None):
    if transform is not None:
        img1, img2 = transform(image=img1)['image'], transform(image=img2)['image']
    img1, img2 = img1 / 255, img2 / 255  # normalize
    # define corners of image patch
    marginal, perturb, patch_size = homo_parameter["marginal"], homo_parameter["perturb"], homo_parameter["patch_size"]
    height, width = homo_parameter["height"], homo_parameter["width"]
    x = random.randint(marginal, width - marginal - patch_size)
    y = random.randint(marginal, height - marginal - patch_size)
    top_left = (x, y)
    bottom_left = (x, patch_size + y - 1)
    bottom_right = (patch_size + x - 1, patch_size + y - 1)
    top_right = (patch_size + x - 1, y)
    four_pts = np.array([top_left, top_right, bottom_left, bottom_right])
    img1 = img1[top_left[1] - marginal:bottom_right[1] + marginal + 1,
           top_left[0] - marginal:bottom_right[0] + marginal + 1, :]
    img2 = img2[top_left[1] - marginal:bottom_right[1] + marginal + 1,
           top_left[0] - marginal:bottom_right[0] + marginal + 1, :]
    four_pts = four_pts - four_pts[np.newaxis, 0]
    (top_left, top_right, bottom_left, bottom_right) = four_pts

    try:
        four_pts_perturb = []
        for i in range(4):
            t1 = random.randint(-perturb, perturb)
            t2 = random.randint(-perturb, perturb)
            four_pts_perturb.append([four_pts[i][0] + t1, four_pts[i][1] + t2])
        org_pts = np.array(four_pts, dtype=np.float32)
        dst_pts = np.array(four_pts_perturb, dtype=np.float32)
        ground_truth = dst_pts - org_pts
        H = cv2.getPerspectiveTransform(org_pts, dst_pts)
        H_inverse = np.linalg.inv(H)
    except:
        four_pts_perturb = []
        for i in range(4):
            t1 = perturb // (i + 1)
            t2 = - perturb // (i + 1)
            four_pts_perturb.append([four_pts[i][0] + t1, four_pts[i][1] + t2])
        org_pts = np.array(four_pts, dtype=np.float32)
        dst_pts = np.array(four_pts_perturb, dtype=np.float32)
        ground_truth = dst_pts - org_pts
        H = cv2.getPerspectiveTransform(org_pts, dst_pts)
        H_inverse = np.linalg.inv(H)

    warped_img1 = cv2.warpPerspective(img1, H_inverse, (img1.shape[1], img1.shape[0]))
    patch_img1 = warped_img1[top_left[1]:bottom_right[1] + 1, top_left[0]:bottom_right[0] + 1, :]
    patch_img2 = img2[top_left[1]:bottom_right[1] + 1, top_left[0]:bottom_right[0] + 1, :]
    patch_img1 = torch.from_numpy(patch_img1).float().permute(2, 0, 1)
    patch_img2 = torch.from_numpy(patch_img2).float().permute(2, 0, 1)
    large_img1 = torch.from_numpy(warped_img1).float().permute(2, 0, 1)
    large_img2 = torch.from_numpy(img2).float().permute(2, 0, 1)

    
    # generate our parameters(HS and HK)
    two_org_pts = np.concatenate((org_pts[:1], org_pts[-1:]))
    two_dst_pts = np.concatenate((dst_pts[:1], dst_pts[-1:]))
    sim = cv2.estimateAffinePartial2D(two_dst_pts, two_org_pts)
    similarity_matrix = np.eye(3)
    similarity_matrix[:2] = sim[0]
    tx = -np.mean(two_dst_pts, axis=0)[0] + (patch_size - 1) / 2
    ty = -np.mean(two_dst_pts, axis=0)[1] + (patch_size - 1) / 2

    H_s3, H_s2_inv = cal_Hs2_Hs2_inv(torch.from_numpy(np.concatenate((two_org_pts.T, np.ones((1, 2))))).unsqueeze(0))

    Hk = H_s3.squeeze(0).double() @ similarity_matrix @ H @ H_s2_inv.squeeze(0).double()
    Hk = Hk / Hk[1, 1]

    a, b, u, v = Hk[0, 0], Hk[0, 2], Hk[0, 1], Hk[2, 1]
    cot_theta1 = (u + b + v + a) * 100
    cot_alpha1 = (-u - b + v + a) * 100
    cot_theta2 = (-u + b - v + a) * 100
    cot_alpha2 = (u - b - v + a) * 100

    angle_gt = np.array([cot_theta1, cot_alpha1, cot_theta2, cot_alpha2]).reshape(1, 2, 2)

    param_gt = np.array([similarity_matrix[0, 0] * 100, similarity_matrix[1, 0] * 100,
                         similarity_matrix[0, 0] * tx - similarity_matrix[1, 0] * ty,
                         similarity_matrix[1, 0] * tx + similarity_matrix[0, 0] * ty,
                         a * 100, b * 100, u * 100, v * 100]).reshape(2, 2, 2)

    return patch_img1, patch_img2, ground_truth, org_pts, dst_pts, param_gt, angle_gt, large_img1, large_img2


def sequence_loss(params_preds, data_batch, args):
    """ Loss function defined over sequence of flow predictions """
    params_gt = data_batch["param_gt"]
    four_gt = data_batch["four_gt"]
    four_preds = calculate_four_pred(params_preds[-1], data_batch)
    mace = ((four_preds - four_gt) ** 2).sum(dim=-1).sqrt().mean(dim=-1).median().detach().cpu()
    mace_ = torch.sum((four_preds - four_gt) ** 2, dim=1).sqrt()

    # for our method we use parameter loss
    loss = loss_function(params_preds, params_gt, data_batch, args)

    Rotation_diff = torch.sum((params_preds[-1][:, :1] - params_gt[:, :1]) ** 2, dim=1).sqrt()
    Kernel_diff = torch.sum((params_preds[-1][:, 1:] - params_gt[:, 1:]) ** 2, dim=1).sqrt()

    metrics = {
        '0.1px': (mace_ < 0.1).float().mean().item(),
        '1px': (mace_ < 1).float().mean().item(),
        '3px': (mace_ < 3).float().mean().item(),

        "R_1": Rotation_diff[:, 0, 0].abs().mean().item(),
        "R_2": Rotation_diff[:, 0, 1].abs().mean().item(),
        "R_3": Rotation_diff[:, 1, 0].abs().mean().item(),
        "R_4": Rotation_diff[:, 1, 1].abs().mean().item(),

        "K_1": Kernel_diff[:, 0, 0].mean().item(),
        "K_2": Kernel_diff[:, 0, 1].mean().item(),
        "K_3": Kernel_diff[:, 1, 0].mean().item(),
        "K_4": Kernel_diff[:, 1, 1].mean().item(),

        'mace': mace.item(),
    }

    return loss, mace, metrics


def loss_function(params_pred, params_gt, data_batch, args):
    loss = 0
    sp_flag = 0
    for i in range(len(params_pred)):
        x = torch.abs(params_pred[i] - params_gt).mean()

        if x < args.speed_threshold: sp_flag = 1
        if args.loss == 'speedupl1':
            loss += (x - sp_flag * (1/(x + args.epsilon)))
        elif args.loss == 'l1':
            loss += x
    return loss


# parameter to Homography matrix
def cal_H(params, coords, downsample=4):
    sim = params[:, :1].flatten(1)
    a0 = sim[:, 0] / 100
    b0 = sim[:, 1] / 100

    H_s0 = torch.eye(3, device=sim.device).repeat(coords.shape[0], 1, 1)
    H_s0[:, 0, 0] = a0
    H_s0[:, 0, 1] = -1 * b0
    H_s0[:, 1, 0] = b0
    H_s0[:, 1, 1] = a0

    H_t1 = torch.eye(3, device=sim.device).repeat(coords.shape[0], 1, 1)
    H_t1[:, 0, 2] = (coords.shape[3] - 1) / 2
    H_t1[:, 1, 2] = (coords.shape[2] - 1) / 2
    H_t1_inv = torch.linalg.inv(H_t1)

    H_t2 = torch.eye(3, device=sim.device).repeat(coords.shape[0], 1, 1)
    H_t2[:, 0, 2] = sim[:, 2] / downsample
    H_t2[:, 1, 2] = sim[:, 3] / downsample
    Hs_1 = H_t1 @ H_t2 @ H_s0 @ H_t1_inv

    kernel = params[:, 1:].flatten(1)
    Hk = torch.eye(3).repeat(coords.shape[0], 1, 1).to(kernel.device)
    Hk[:, 0, 0] = kernel[:, 0] / 100
    Hk[:, 2, 2] = kernel[:, 0] / 100
    Hk[:, 0, 2] = kernel[:, 1] / 100
    Hk[:, 2, 0] = kernel[:, 1] / 100
    Hk[:, 0, 1] = kernel[:, 2] / 100
    Hk[:, 2, 1] = kernel[:, 3] / 100

    two_point_org_MN = torch.zeros((1, 2, 2), device=kernel.device)

    two_point_org_MN[:, 1, :] = torch.Tensor([0, coords.shape[2] - 1])
    two_point_org_MN[:, 0, :] = torch.Tensor([0, coords.shape[3] - 1])
    H_s3, H_s2_inv = cal_Hs2_Hs2_inv(two_point_org_MN[:, :2])

    H = torch.linalg.inv(Hs_1) @ H_s2_inv @ Hk @ H_s3

    return H


def cal_Hs2_Hs2_inv(points_MN):
    O_x = (points_MN[:, 0, 0] + points_MN[:, 0, 1]) / 2
    O_y = (points_MN[:, 1, 0] + points_MN[:, 1, 1]) / 2
    Matrix_1 = torch.eye(3, dtype=torch.float32, device=points_MN.device).repeat(points_MN.shape[0], 1, 1)
    Matrix_1[:, 0, 2] = -O_x
    Matrix_1[:, 1, 2] = -O_y

    ON_x = points_MN[:, 0, 1] - O_x
    ON_y = points_MN[:, 1, 1] - O_y
    as1 = ON_x / (ON_x ** 2 + ON_y ** 2)
    bs1 = -ON_y / (ON_x ** 2 + ON_y ** 2)
    Matrix_2 = torch.eye(3, dtype=torch.float32, device=points_MN.device).repeat(points_MN.shape[0], 1, 1)
    Matrix_2[:, 0, 0] = as1
    Matrix_2[:, 1, 1] = as1
    Matrix_2[:, 0, 1] = -bs1
    Matrix_2[:, 1, 0] = bs1

    Hs2 = Matrix_2 @ Matrix_1
    Hs2_inv = torch.eye(3, dtype=torch.float32, device=points_MN.device).repeat(points_MN.shape[0], 1, 1)
    Hs2_inv[:, 0, 0] = ON_x
    Hs2_inv[:, 1, 1] = ON_x
    Hs2_inv[:, 0, 1] = -ON_y
    Hs2_inv[:, 1, 0] = ON_y
    Hs2_inv[:, 0, 2] = O_x
    Hs2_inv[:, 1, 2] = O_y

    return Hs2, Hs2_inv

# parameters to four disp
def calculate_four_pred(params_preds, data_batch):
    coords = torch.ones((params_preds.shape[0], 2, 128, 128), device=params_preds.device)
    H_pred = cal_H(params_preds, coords, 1)
    ones = torch.ones((params_preds.shape[0], 4, 1), device=params_preds.device)
    four_pts_homogeneous = torch.cat((data_batch["org_pts"], ones), dim=2)
    four_pts_homogeneous = four_pts_homogeneous.permute(0, 2, 1)
    four_pred = H_pred @ four_pts_homogeneous
    four_pred = four_pred / four_pred[:, -1:]

    return four_pred[:, :-1].permute(0, 2, 1) - data_batch["org_pts"]

# parameters to angular
def calculate_four_cot(params_preds):
    Hk_param = params_preds[:, 1:].flatten(1)
    a, b, u, v = Hk_param[:, 0], Hk_param[:, 1], Hk_param[:, 2], Hk_param[:, 3]
    arctan_theta1 = u + b + v + a
    arctan_alpha1 = -u - b + v + a
    arctan_theta2 = -u + b - v + a
    arctan_alpha2 = u - b - v + a

    angle_pred = torch.stack([arctan_theta1, arctan_alpha1, arctan_theta2, arctan_alpha2], dim=1).reshape(-1, 1, 2, 2)

    return angle_pred

