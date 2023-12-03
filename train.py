# -*-codeing = utf-8 -*-
# @Time :2023/4/14 20:39
# @Author: zhu
# @Site:
# @File:train.py
# @Software:PyCharm
import torch
from torch import nn
from Mult import Mult

from src.utils import *

import time


from torch.utils.data import DataLoader
from src.eval_metrics import *
import argparse
import matplotlib.pyplot as plt
parser = argparse.ArgumentParser(description='MOSEI Sentiment Analysis')
parser.add_argument('-f', default='', type=str)

# Fixed
parser.add_argument('--model', type=str, default='MulT',
                    help='name of the model to use (Transformer, etc.)')

# Tasks
parser.add_argument('--vonly', action='store_true',
                    help='use the crossmodal fusion into v (default: False)')
parser.add_argument('--aonly', action='store_true',
                    help='use the crossmodal fusion into a (default: False)')
parser.add_argument('--lonly', action='store_true',
                    help='use the crossmodal fusion into l (default: False)')
parser.add_argument('--aligned', action='store_true',
                    help='consider aligned experiment or not (default: False)')
parser.add_argument('--dataset', type=str, default='mosei_senti',
                    help='dataset to use (default: mosei_senti)')
parser.add_argument('--data_path', type=str, default=r'C:\Users\24525\Desktop\MSA\dataset\CMU-MOSEI\MultDataset',
                    help='path for storing the dataset')

# Dropouts
parser.add_argument('--attn_dropout', type=float, default=0.1,
                    help='attention dropout')
parser.add_argument('--attn_dropout_a', type=float, default=0.0,
                    help='attention dropout (for audio)')
parser.add_argument('--attn_dropout_v', type=float, default=0.0,
                    help='attention dropout (for visual)')
parser.add_argument('--relu_dropout', type=float, default=0.1,
                    help='relu dropout')
parser.add_argument('--embed_dropout', type=float, default=0.25,
                    help='embedding dropout')
parser.add_argument('--res_dropout', type=float, default=0.1,
                    help='residual block dropout')
parser.add_argument('--out_dropout', type=float, default=0.0,
                    help='output layer dropout')

# Architecture
parser.add_argument('--nlevels', type=int, default=5,
                    help='number of layers in the network (default: 5)')
parser.add_argument('--num_heads', type=int, default=5,
                    help='number of heads for the transformer network (default: 5)')
parser.add_argument('--attn_mask', action='store_false',
                    help='use attention mask for Transformer (default: true)')

# Tuning
parser.add_argument('--batch_size', type=int, default=8, metavar='N',
                    help='batch size (default: 8)')
parser.add_argument('--clip', type=float, default=0.8,
                    help='gradient clip value (default: 0.8)')
parser.add_argument('--lr', type=float, default=1e-3,
                    help='initial learning rate (default: 1e-3)')
parser.add_argument('--optim', type=str, default='Adam',
                    help='optimizer to use (default: Adam)')
parser.add_argument('--num_epochs', type=int, default=40,
                    help='number of epochs (default: 40)')
parser.add_argument('--when', type=int, default=20,
                    help='when to decay learning rate (default: 20)')
parser.add_argument('--batch_chunk', type=int, default=1,
                    help='number of chunks per batch (default: 1)')

# Logistics
parser.add_argument('--log_interval', type=int, default=30,
                    help='frequency of result logging (default: 30)')
parser.add_argument('--seed', type=int, default=1111,
                    help='random seed')
parser.add_argument('--no_cuda', action='store_true',
                    help='do not use cuda')
parser.add_argument('--name', type=str, default='mult',
                    help='name of the trial (default: "mult")')
args = parser.parse_args()

torch.manual_seed(args.seed)
dataset = str.lower(args.dataset.strip())

print("Start loading the data....")

train_data = get_data(args, dataset, 'train')
valid_data = get_data(args, dataset, 'valid')
test_data = get_data(args, dataset, 'test')

train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True,
                          generator=torch.Generator(device='cuda'))
valid_loader = DataLoader(valid_data, batch_size=args.batch_size, shuffle=True,
                          generator=torch.Generator(device='cuda'))
test_loader = DataLoader(test_data, batch_size=args.batch_size, shuffle=True, generator=torch.Generator(device='cuda'))

print('Finish loading the data....')
if not args.aligned:
    print("### Note: You are running in unaligned mode.")

def train(model):
    epoch_loss = 0
    model.train()
    num_batches = len(train_data) // 8
    proc_loss, proc_size = 0, 0
    start_time = time.time()
    # 定义L1损失
    criterion = nn.L1Loss()
    clip = 0.8
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    loss_his = []
    for i_batch, (batch_X, batch_Y, batch_META) in enumerate(train_loader):
        sample_ind, text, audio, vision = batch_X
        eval_attr = batch_Y.squeeze(-1)  # if num of labels is 1

        with torch.cuda.device(0):
            text, audio, vision, eval_attr = text.cuda(), audio.cuda(), vision.cuda(), eval_attr.cuda()

        batch_size = text.size(0)
        # batch_chunk = 1

        # combined_loss = 0
        net = nn.DataParallel(model) if batch_size > 10 else model

        preds, hiddens = net(text, audio, vision)
        raw_loss = criterion(preds, eval_attr)
        raw_loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        optimizer.step()

        proc_loss += raw_loss.item() * batch_size
        proc_size += batch_size
        # epoch_loss += combined_loss.item() * batch_size
        if i_batch % 30 == 0 and i_batch > 0:
            avg_loss = proc_loss / proc_size
            elapsed_time = time.time() - start_time
            print('Epoch {:2d} | Batch {:3d}/{:3d} | Time/Batch(ms) {:5.2f} | Train Loss {:5.4f}'.
                  format(epoch, i_batch, num_batches, elapsed_time * 1000 / 30, avg_loss))
            loss_his.append(avg_loss)
            proc_loss, proc_size = 0, 0
            start_time = time.time()
    # 画图
    # plt.plot(loss_his)
    # plt.show()

    return epoch_loss / len(train_loader)

def evaluate(model, test=False):
    model.eval()
    loader = test_loader if test else valid_loader
    total_loss = 0.0

    results = []
    truths = []
    criterion = nn.L1Loss()
    with torch.no_grad():
        for i_batch, (batch_X, batch_Y, batch_META) in enumerate(loader):
            sample_ind, text, audio, vision = batch_X
            eval_attr = batch_Y.squeeze(dim=-1)  # if num of labels is 1
            with torch.cuda.device(0):
                text, audio, vision, eval_attr = text.cuda(), audio.cuda(), vision.cuda(), eval_attr.cuda()


            batch_size = text.size(0)

            net = nn.DataParallel(model) if batch_size > 10 else model
            preds, _ = net(text, audio, vision)
            total_loss += criterion(preds, eval_attr).item() * batch_size

            # Collect the results into dictionary
            results.append(preds)
            truths.append(eval_attr)

    avg_loss = total_loss / (len(test_loader)if test else len(valid_loader))

    results = torch.cat(results)
    truths = torch.cat(truths)
    return avg_loss, results, truths


# model = Mult(train_data.text.shape[-1],train_data.audio.shape[-1],train_data.vision.shape[-1])
try:
    model = torch.load(f'pre_trained_models/myMult.pt')
except:
    model = Mult(train_data.text.shape[-1], train_data.audio.shape[-1], train_data.vision.shape[-1])

best_valid = 1e8
num_epochs = 40
optimizer_1 = torch.optim.Adam(model.parameters(), lr=1e-3)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer_1, mode='min', factor=0.5, patience=5, verbose=True)
for epoch in range(1, num_epochs + 1):
    start = time.time()
    train(model)
    val_loss, _, _ = evaluate(model, test=False)
    test_loss, _, _ = evaluate(model,test=True)

    end = time.time()
    duration = end - start
    scheduler.step(val_loss)  # Decay learning rate by validation loss

    print("-" * 50)
    print(
        'Epoch {:2d} | Time {:5.4f} sec | Valid Loss {:5.4f} | Test Loss {:5.4f}'.format(epoch, duration, val_loss,
                                                                                         test_loss))
    print("-" * 50)

    if val_loss < best_valid:
        # print(f"Saved model at pre_trained_models/{hyp_params.name}.pt!")
        torch.save(model, f'pre_trained_models/myMult.pt')
        best_valid = val_loss

_, results, truths = evaluate(model, test=True)

eval_mosei_senti(results, truths, True)