import os
import time
from tqdm import tqdm
import numpy as np
import torch
from torch.autograd import Variable
import torch.nn as nn

def trainepoch(writer, steps, nli_net, loss_fn, optimizer, scheduler, params, epoch, train_dataloader, len_train_dataset):
    print('\nTRAINING : Epoch ' + str(epoch))
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    nli_net.train()
    all_costs = []
    logs = []
    words_count = 0

    last_time = time.time()
    correct = 0.

    # optimizer.param_groups[0]['lr'] = optimizer.param_groups[0]['lr'] * params.decay if epoch>1\
    #     and 'sgd' in params.optimizer else optimizer.param_groups[0]['lr']

    print('Learning rate : {0}'.format(optimizer.param_groups[0]['lr']))
    writer.add_scalar("Learning Rate", optimizer.param_groups[0]['lr'], epoch)

    for stidx, train_batch in enumerate(tqdm(train_dataloader)):
        s1_batch, s2_batch, train_labels, s1_len, s2_len = train_batch["s1_batch"], train_batch["s2_batch"], train_batch["target"], train_batch["s1_len"], train_batch["s2_len"]

        s1_batch, s2_batch = Variable(s1_batch.to(device)), Variable(s2_batch.to(device))
        tgt_batch = Variable(torch.LongTensor(train_labels)).to(device)
        k = s1_batch.size(1)  # actual batch size

        # model forward
        output = nli_net((s1_batch, s1_len.to(device)), (s2_batch, s2_len.to(device)))

        pred = output.data.max(1)[1]
        correct += pred.long().eq(tgt_batch.data.long()).cpu().sum()

        # loss
        loss = loss_fn(output, tgt_batch)
        all_costs.append(loss.data)
        words_count += (s1_batch.nelement() + s2_batch.nelement()) / params.word_emb_dim

        # backward
        optimizer.zero_grad()
        loss.backward()

        # gradient clipping (off by default)
        shrink_factor = 1
        total_norm = 0

        for p in nli_net.parameters():
            if p.requires_grad:
                p.grad.data.div_(k)  # divide by the actual batch size
                total_norm += p.grad.data.norm() ** 2
        total_norm = torch.sqrt(total_norm)

        if total_norm > params.max_norm:
            shrink_factor = params.max_norm / total_norm
        current_lr = optimizer.param_groups[0]['lr'] # current lr (no external "lr", for adam)
        optimizer.param_groups[0]['lr'] = current_lr * shrink_factor # just for update

        # optimizer step
        optimizer.step()
        optimizer.param_groups[0]['lr'] = current_lr

        steps += 1
        if len(all_costs) == 100:
            logs = '{0} ; loss {1} ; sentence/s {2} ; words/s {3} ; accuracy train : {4}'.format(
                            stidx, np.round(torch.mean(torch.stack(all_costs)).detach().cpu().numpy(), 2),
                            int(len(all_costs) * params.batch_size / (time.time() - last_time)),
                            int(words_count * 1.0 / (time.time() - last_time)),
                            round(100.*np.array(correct)/(stidx+k), 2))
            # print(logs[-1])
            print(logs)
            writer.add_scalar("Training Loss", np.round(torch.mean(torch.stack(all_costs)).detach().cpu().numpy(), 2), steps)
            last_time = time.time()
            words_count = 0
            all_costs = []

    scheduler.step()
    train_acc = round(100 * np.array(correct)/len_train_dataset, 2)
    print('results : epoch {0} ; mean accuracy train : {1}'
          .format(epoch, train_acc))
    writer.add_scalar("Training Acc", train_acc, epoch)
    return train_acc, steps


def evaluate(writer, nli_net, optimizer, params, val_acc_best, stop_training, epoch, valid_dataloader, len_val, eval_type='valid', final_eval=False):
    nli_net.eval()
    correct = 0.
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    if eval_type == 'valid':
        print('\nVALIDATION : Epoch {0}'.format(epoch))

    for stidx, val_batch in enumerate(tqdm(valid_dataloader)):
        s1_batch, s2_batch, val_labels, s1_len, s2_len = val_batch["s1_batch"], val_batch["s2_batch"], val_batch["target"], val_batch["s1_len"], val_batch["s2_len"]
        # prepare batch
        s1_batch, s2_batch = Variable(s1_batch.to(device)), Variable(s2_batch.to(device))
        tgt_batch = Variable(val_labels).to(device)

        # model forward
        output = nli_net((s1_batch, s1_len.to(device)), (s2_batch, s2_len.to(device)))

        pred = output.data.max(1)[1]
        correct += pred.long().eq(tgt_batch.data.long()).cpu().sum()

    # save model
    eval_acc = round(100 * np.array(correct) / len_val, 2)
    if final_eval:
        print('finalgrep : accuracy {0} : {1}'.format(eval_type, eval_acc))
    else:
        print('togrep : results : epoch {0} ; mean accuracy {1} :\
              {2}'.format(epoch, eval_type, eval_acc))

    if eval_type == 'valid' and epoch <= params.n_epochs:
        if eval_acc > val_acc_best:
            print('saving model at epoch {0}'.format(epoch))
            if not os.path.exists(params.outputdir):
                os.makedirs(params.outputdir)
            torch.save(nli_net, os.path.join(params.outputdir, params.outputmodelname))
            val_acc_best = eval_acc
        else:
            optimizer.param_groups[0]['lr'] = optimizer.param_groups[0]['lr'] / params.lrshrink
            print('Shrinking lr by : {0}. New lr = {1}'.format(params.lrshrink, optimizer.param_groups[0]['lr']))
            if optimizer.param_groups[0]['lr'] < params.minlr:
                stop_training = True
    
    if writer is not None:
        writer.add_scalar("{} Acc".format(eval_type), eval_acc, epoch)
    return eval_acc, stop_training
