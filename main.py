import os, torch, logging, argparse
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt

from tqdm import tqdm

from dataset import get_dataset
from model import SRCNN, DRRN, Net
from loss import FMSELoss
from utils import AverageMeter, calc_psnr, interpolation


def get_modules(args):
    model = Net().double()
    criterion = FMSELoss()
    optimizer = optim.SGD(model.parameters(), lr=args.lr)
    return model, criterion, optimizer

def train(args):
    # initialize model 
    model, criterion, optimizer = get_modules(args)

    # load dataset
    dataset = get_dataset(args.data_root, 'train', args.lq_scale)  
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)

    # train
    for epoch in range(args.num_epochs):
        model.train()
        epoch_losses = AverageMeter()

        # iterate with batch
        with tqdm(total=(len(dataset) - len(dataset) % args.batch_size)) as t:
            t.set_description('epoch: {}/{}'.format(epoch, args.num_epochs - 1))
            for i, data in enumerate(dataloader):
                # if i > 2:
                #     break
                depth, depth_lq, sidescan = data

                # forward pass
                preds = model(depth_lq.double(), sidescan.double(), args)
                loss = criterion(preds, depth)
                epoch_losses.update(loss.item(), len(depth_lq))

                # back prop
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                # update status
                t.set_postfix(loss='{:.6f}'.format(epoch_losses.avg))
                t.update(len(depth_lq))

        torch.save(model.state_dict(), os.path.join(args.outputs_dir, 'epoch_{}.pth'.format(epoch)))
        torch.save(model.state_dict(), os.path.join(args.outputs_dir, 'model_last.pth'))

def test(args):
    # initialize measurement
    epoch_psnr = AverageMeter()
    inters_psnr = AverageMeter()

    # load dataset
    dataset = get_dataset(args.data_root, 'test', args.lq_scale)  
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)

    # load trained model
    model, _, _ = get_modules(args)
    model.load_state_dict(torch.load(os.path.join(args.outputs_dir, 'model_last.pth')))

    # iterate with batch
    with tqdm(total=(len(dataset) - len(dataset) % args.batch_size)) as t:
        for i, data in enumerate(dataloader):
            depth, depth_lq, sidescan = data
            with torch.no_grad():
                # preds = model(depth_lq).double().clamp(0.0, 1.0)
                preds = model(depth_lq.double(), sidescan.double(), args)
                inters = interpolation(depth_lq.double(), args.batch_size, args.lq_scale).type(torch.double)
            
            epoch_psnr.update(calc_psnr(preds, depth), len(depth_lq))
            inters_psnr.update(calc_psnr(inters, depth), len(depth_lq))

    print('interpolation psnr: {:.2f}'.format(inters_psnr.avg))
    print('eval psnr: {:.2f}'.format(epoch_psnr.avg))

def plot(args):    
    # load dataset
    dataset = get_dataset(args.data_root, 'test', args.lq_scale)  
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)

    # load trained model
    model, _, _ = get_modules(args)
    model.load_state_dict(torch.load(os.path.join(args.outputs_dir, 'model_last.pth')))

    # iterate and calc score
    for i, data in enumerate(dataloader):
        depth, depth_lq, sidescan = data
        with torch.no_grad():
            # preds = model(depth_lq).double().clamp(0.0, 1.0)
            preds = model(depth_lq.double(), sidescan.double(), args)
            inters = interpolation(depth_lq.double(), args.batch_size, args.lq_scale).type(torch.double)
        for j in range(args.batch_size):
            fig, ax = plt.subplots(1, 5)

            ax[0].imshow(sidescan[j].reshape(256, 256))
            ax[0].set_title("Sidescan")
 
            ax[1].imshow(depth_lq[j].reshape(256, 256))
            ax[1].set_title("Depth Low Quality")
 
            ax[2].imshow(inters[j].reshape(256, 256))
            ax[2].set_title("Interpolation")

            ax[3].imshow(preds[j].reshape(256, 256))
            ax[3].set_title("Prediction")

            ax[4].imshow(depth[j].reshape(256, 256))
            ax[4].set_title("Depth Ground Truth")

            plt.show()


if __name__ == "__main__":
    # set argument parser
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-root', type=str, default='./sss2depth_split')
    parser.add_argument('--outputs-dir', type=str, default='./trained')
    parser.add_argument('--lq-scale', type=int, default=32)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--batch-size', type=int, default=4)
    parser.add_argument('--num-epochs', type=int, default=5)
    parser.add_argument('--seed', type=int, default=123)
    args = parser.parse_args()
    
    train(args)
    test(args)
    #plot(args)

