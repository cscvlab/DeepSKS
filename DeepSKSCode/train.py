import argparse
import pprint
import time
import warnings

warnings.filterwarnings("ignore")
import datasets
from network import *
from utils import *
from homo_utils import *


def train(model, train_loader, optimizer, scheduler, logger, args):
    for i, data_batch in enumerate(train_loader):
        tic = time.time()
        for key, value in data_batch.items():
            if type(data_batch[key]) == torch.Tensor: data_batch[key] = data_batch[key].to(torch.device('cuda:' + str(args.gpuid)))

        optimizer.zero_grad()
        pred_h4p_12 = model(data_batch)
        loss, mace, metrics = sequence_loss(pred_h4p_12, data_batch, args)

        loss.backward()
        optimizer.step()
        scheduler.step()

        toc = time.time()
        metrics['time'] = toc - tic
        logger.push(metrics)

        if logger.total_steps % args.val_freq == args.val_freq - 1:
            plot_train(logger, args)
            PATH = args.output + f'/{logger.total_steps + 1}_{args.name}.pth'
            checkpoint = {
                "net": model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict(),
            }
            torch.save(checkpoint, PATH)
        if logger.total_steps >= args.num_steps:
            break


def main(args):
    device = torch.device('cuda:' + str(args.gpuid))
    train_loader = datasets.fetch_dataloader(args, split="train")
    homo_model = MCNet(args).to(device)
    homo_model.train()
    optimizer, scheduler = fetch_optimizer(args, homo_model)

    if args.restore_ckpt is not None:
        save_model = torch.load(args.restore_ckpt)
        homo_model.load_state_dict(save_model['net'])
        optimizer.load_state_dict(save_model['optimizer'])
        scheduler.load_state_dict(save_model['scheduler'])

    print(f"Parameter Count: {count_parameters(homo_model)}")

    logger = Logger(homo_model, scheduler, args)
    logger.total_steps = scheduler.last_epoch

    while logger.total_steps <= args.num_steps:
        train(homo_model, train_loader, optimizer, scheduler, logger, args)
        if logger.total_steps >= args.num_steps:
            plot_train(logger, args)
            break

    PATH = args.output + f'/{args.name}.pth'
    torch.save(homo_model.state_dict(), PATH)

    return PATH



if __name__ == "__main__":
    dataset_names = "MSCOCO"
    model_name = "MCNet_param"
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', default=f'{model_name}', help="name your experiment")
    parser.add_argument('--gpuid', type=int, default=0)
    parser.add_argument('--note', type=str, default=f'{model_name}', help='experiment notes')
    parser.add_argument('--restore_ckpt', help="restore checkpoint")
    parser.add_argument('--dataset', type=str, default=dataset_names, help='dataset')
    parser.add_argument('--log_dir', type=str, default=f'logs/{dataset_names}', help='The log path')
    parser.add_argument('--nolog', action='store_true', default=False, help='save log file or not')
    parser.add_argument('--output', type=str, default=f'Results/{dataset_names}/{model_name}',
                        help='output directory to save checkpoints and plots')
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--print_freq', type=int, default=100)
    parser.add_argument('--val_freq', type=int, default=20000)
    parser.add_argument('--seed', type=int, default=512)
    parser.add_argument('--num_steps', type=int, default=120000)
    parser.add_argument('--lr', type=float, default=4e-4, help='Max learning rate')
    parser.add_argument('--log_full_dir', type=str)
    parser.add_argument('--epsilon', type=float, default=1, help='loss parameter')
    parser.add_argument('--wdecay', type=float, default=0.00001)
    parser.add_argument('--loss', type=str, default="l1", help="l1 or speedupl1")
    parser.add_argument('--downsample', type=int, nargs='+', default=[4, 2, 2])
    parser.add_argument('--iter', type=int, nargs='+', default=[2, 2, 2])
    parser.add_argument('--speed_threshold', type=float, default=10, help='use speed-up when L1 < x')
    args = parser.parse_args()

    if not args.nolog:
        args.log_full_dir = os.path.join(args.log_dir, time.strftime("%Y-%m-%d-%H_%M_%S",
                                                                     time.localtime()) + "_" + args.dataset + "_" + args.note)
        if not os.path.exists(args.log_full_dir): os.makedirs(args.log_full_dir)
        sys.stdout = Logger_(os.path.join(args.log_full_dir, f'record.log'), sys.stdout)
    if not os.path.isdir(args.output): os.makedirs(args.output)
    pprint.pprint(vars(args))

    seed_everything(args.seed)

    main(args)
