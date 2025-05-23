import argparse
import pprint
import time
import warnings

import scipy

warnings.filterwarnings("ignore")
import datasets
from network import *
from utils import *
from homo_utils import *


def test(args, glob_iter=None, homo_model=None):
    device = torch.device('cuda:' + str(args.gpuid))
    test_loader = datasets.fetch_dataloader(args, split="test")
    if homo_model == None:
        homo_model = MCNet(args).to(device)
        if args.checkpoint is None:
            print("ERROR : no checkpoint")
            exit()
        state = torch.load(args.checkpoint, map_location='cpu')
        homo_model.load_state_dict(state)
        print("test with pretrained model")
    homo_model.eval()

    with torch.no_grad():
        mace_offset = np.array([])
        mace_angular = np.array([])
        for test_repeat in range(1):
            for i, data_batch in enumerate(test_loader):
                for key, value in data_batch.items():
                    if type(data_batch[key]) == torch.Tensor: data_batch[key] = data_batch[key].to(device)

                pred_h4p_12 = homo_model(data_batch)

                # calculate angular metric
                four_cot_preds = calculate_four_cot(pred_h4p_12[-1]) / 100
                four_cot_gt = data_batch["angle_gt"] / 100
                cot_mace = ((four_cot_preds - four_cot_gt) ** 2).reshape(1, -1).sqrt().mean(dim=-1).detach().cpu().numpy()
                mace_angular = np.concatenate([mace_angular, cot_mace])

                # calculate offset metric
                four_preds = calculate_four_pred(pred_h4p_12[-1], data_batch)
                mace = ((four_preds - data_batch["four_gt"]) ** 2).sum(dim=-1).sqrt().mean(
                    dim=-1).detach().cpu().numpy()
                mace_offset = np.concatenate([mace_offset, mace])

                print(f"Number:{i}, offset:{round(mace[0], 4)}, angular:{round(cot_mace[0], 4)}")


    # mean and median of offset_metric and angular_metric
    print(f"offset_mean:{round(np.mean(mace_offset), 5)}")
    print(f"offset_median:{round(np.median(mace_offset), 5)}")

    print(f"angular_mean:{round(np.mean(mace_angular), 5)}")
    print(f"angular_median:{round(np.median(mace_angular), 5)}")

    if not os.path.exists("res_mat"):
        os.makedirs("res_mat")
    if args.save:
        scipy.io.savemat('res_mat/' + args.savemat_angular, {'matrix': mace_angular})
        scipy.io.savemat('res_mat/' + args.savemat_offset, {'matrix': mace_offset})





def main():
    dataset_names = "MSCOCO" # ["MSCOCO", "SPID", "GoogleMap"]
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, default='test', help='Train or test', choices=['train', 'test'])
    parser.add_argument('--gpuid', type=int, default=0)
    parser.add_argument('--save', type=bool, default=False, help="save metric mat or not")
    parser.add_argument('--savemat_offset', type=str, default=f'{dataset_names}/MCNet_param_geometric_offset.mat')
    parser.add_argument('--savemat_angular', type=str, default=f'{dataset_names}/MCNet_param_geometric_angular.mat')
    parser.add_argument('--dataset', type=str, default=dataset_names, help='dataset')
    parser.add_argument('--checkpoint', type=str,
                        default=f"Results/{dataset_names}/MCNet_param_geometric/MCNet_param_geometric.pth",
                        help='Test model name')
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--seed', type=int, default=512)
    parser.add_argument('--downsample', type=int, nargs='+', default=[4, 2, 1])
    parser.add_argument('--iter', type=int, nargs='+', default=[2, 2, 2])
    args = parser.parse_args()

    seed_everything(args.seed)

    test(args)


if __name__ == "__main__":
    main()
