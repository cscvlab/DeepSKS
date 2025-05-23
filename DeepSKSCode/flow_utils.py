import torch
import torchgeometry as tgm
import torch.nn.functional as F
from homo_utils import cal_H

def bilinear_sampler(img, coords, mode='bilinear', mask=False):
    """ Wrapper for grid_sample, uses pixel coordinates """
    H, W = img.shape[-2:]
    xgrid, ygrid = coords.split([1,1], dim=-1)
    xgrid = 2*xgrid/(W-1) - 1
    ygrid = 2*ygrid/(H-1) - 1

    grid = torch.cat([xgrid, ygrid], dim=-1)
    img = F.grid_sample(img, grid, align_corners=True)

    if mask:
        mask = (xgrid > -1) & (ygrid > -1) & (xgrid < 1) & (ygrid < 1)
        return img, mask.float()

    return img


def coords_grid(batch, ht, wd):
    coords = torch.meshgrid(torch.arange(ht), torch.arange(wd))
    coords = torch.stack(coords[::-1], dim=0).float()
    return coords[None].expand(batch, -1, -1, -1)


def initialize_flow(img, downsample=4):
    N, C, H, W = img.shape
    coords0 = coords_grid(N, H//downsample, W//downsample).to(img.device)
    coords1 = coords_grid(N, H//downsample, W//downsample).to(img.device)

    return coords0, coords1

# our method to get disp_to_coords
def disp_to_coords_ours(params, coords, data_batch, downsample=4):
    H = cal_H(params, coords, downsample=downsample)

    gridy, gridx = torch.meshgrid(torch.linspace(0, coords.shape[3] - 1, steps=coords.shape[3]),
                                  torch.linspace(0, coords.shape[2] - 1, steps=coords.shape[2]))
    points = torch.cat((gridx.flatten().unsqueeze(0), gridy.flatten().unsqueeze(0),
                        torch.ones((1, coords.shape[3] * coords.shape[2]))),
                       dim=0).unsqueeze(0).repeat(coords.shape[0], 1, 1).to(params.device)
    points_new = H.bmm(points)
    points_new = points_new / (points_new[:, 2, :].unsqueeze(1) + torch.tensor(1e-5))
    points_new = points_new[:, 0:2, :]
    coords = torch.cat((points_new[:, 0, :].reshape(coords.shape[0], coords.shape[3], coords.shape[2]).unsqueeze(1),
                        points_new[:, 1, :].reshape(coords.shape[0], coords.shape[3], coords.shape[2]).unsqueeze(1)),
                       dim=1)
    return coords
