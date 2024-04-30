import os
import sys
import time
import argparse
import numpy as np
import torch
import torch.nn as nn
# from mutils import get_optimizer
from torch import optim
import torch.optim.lr_scheduler as lr_scheduler
from new_models import NLINet
from new_dataset import SNLIDataset, collate_fn, get_nli, build_vocab, load_vocab, save_vocab
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime as d
from new_train import trainepoch, evaluate


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='NLI training')
    parser.add_argument("--nlipath", type=str, default='dataset/SNLI/', help="NLI data path (SNLI or MultiNLI)")
    parser.add_argument("--outputdir", type=str, default='savedir2/', help="Output directory")
    parser.add_argument("--logdir", type=str, default='logdir/', help="Logs directory")
    parser.add_argument("--outputmodelname", type=str, default='model.pickle')
    parser.add_argument("--vocab_path", type=str, default='vocab.pickle')
    parser.add_argument("--n_epochs", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--dpout_model", type=float, default=0., help="encoder dropout")
    parser.add_argument("--dpout_fc", type=float, default=0., help="classifier dropout")
    parser.add_argument("--nonlinear_fc", type=float, default=1, help="use nonlinearity in fc")
    parser.add_argument("--lr", type=float, default=0.1, help="shrink factor for sgd")
    parser.add_argument("--lrshrink", type=float, default=5, help="shrink factor for sgd")
    parser.add_argument("--decay", type=float, default=0.99, help="lr decay")
    parser.add_argument("--minlr", type=float, default=10e-5, help="minimum lr")
    parser.add_argument("--max_norm", type=float, default=5., help="max norm (grad clipping)")
    parser.add_argument("--encoder_type", type=str, default='BLSTMEncoder', choices=['AWESentenceEncoder', 'BLSTMEncoder', 'LSTMEncoder'], help="see list of encoders")
    parser.add_argument("--enc_lstm_dim", type=int, default=2048, help="encoder nhid dimension")
    parser.add_argument("--n_enc_layers", type=int, default=1, help="encoder num layers")
    parser.add_argument("--fc_dim", type=int, default=512, help="nhid of fc layers")
    parser.add_argument("--word_emb_dim", type=int, default=300, help="word embedding dimension")
    parser.add_argument("--n_classes", type=int, default=3, help="entailment/neutral/contradiction")
    parser.add_argument("--pool_type", type=str, default='last', choices=['max', 'last'], help="max or mean")
    parser.add_argument("--gpu_id", type=int, default=3, help="GPU ID")
    parser.add_argument("--seed", type=int, default=1234, help="seed")
    params, _ = parser.parse_known_args()

    params.outputmodelname = params.encoder_type + '_' + params.pool_type + '_' + params.outputmodelname
    logdir = params.logdir
    logdir = os.path.join(logdir, params.encoder_type + '_' + params.pool_type, f"{d.now().strftime('%Y-%m-%d-%H-%M-%S')}")
    writer = SummaryWriter(logdir)

    # set gpu device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    isCuda = True if device == 'cuda' else False

    # print parameters passed, and all parameters
    print('\ntogrep : {0}\n'.format(sys.argv[1:]))
    print(params)

    # Seed
    np.random.seed(params.seed)
    torch.manual_seed(params.seed)
    if isCuda:
        torch.cuda.manual_seed(params.seed)

    # Create vocabulary
    train, valid, test = get_nli()
    sentences = train['premise'] + train['hypothesis'] + \
    valid['premise'] + valid['hypothesis'] + \
    test['premise'] + test['hypothesis']

    start_time = time.time()
    word_vec = build_vocab(sentences)
    print('Time to build vocab: {0}'.format(time.time() - start_time))

    # Model
    nli_net = NLINet(params)
    print(nli_net)

    # Loss
    weight = torch.FloatTensor(params.n_classes).fill_(1)
    loss_fn = nn.CrossEntropyLoss(weight=weight)
    loss_fn.size_average = False

    # Optimizer
    # optim_fn, optim_params = get_optimizer(params.optimizer)
    # optimizer = optim_fn(nli_net.parameters(), **optim_params)
    optimizer = optim.SGD(nli_net.parameters(), lr = params.lr)
    scheduler = lr_scheduler.MultiplicativeLR(optimizer, lr_lambda=lambda epoch: params.decay, verbose = True)

    if isCuda:
        nli_net.cuda()
        loss_fn.cuda()

    # Data loaders
    train_dataset = SNLIDataset(train['premise'], train['hypothesis'], train['label'], word_vec)
    train_dataloader = DataLoader(train_dataset, batch_size= params.batch_size, collate_fn=collate_fn)
    valid_dataset = SNLIDataset(valid['premise'], valid['hypothesis'], valid['label'], word_vec)
    valid_dataloader = DataLoader(valid_dataset, batch_size=64, collate_fn=collate_fn)
    test_dataset = SNLIDataset(test['premise'], test['hypothesis'], test['label'], word_vec)
    test_dataloader = DataLoader(test_dataset, batch_size=64, collate_fn=collate_fn)

    # Start training ...
    epoch = 1
    steps = 1
    val_acc_best = -1e10
    stop_training = False
    while not stop_training and epoch <= params.n_epochs:
        train_acc, steps = trainepoch(writer, steps, nli_net, loss_fn, optimizer, scheduler, params, epoch, train_dataloader, len(train_dataset))
        eval_acc, stop_training = evaluate(writer, nli_net, optimizer, params, val_acc_best, stop_training, epoch, valid_dataloader, len(valid_dataset), 'valid')
        epoch += 1

    # Test on the best model ...
    del nli_net
    print(os.path.join(params.outputdir, params.outputmodelname))
    nli_net = torch.load(os.path.join(params.outputdir, params.outputmodelname))

    print('\nTEST : Epoch {0}'.format(epoch))
    evaluate(writer, nli_net, optimizer, params, val_acc_best, stop_training, 1e6, valid_dataloader, len(valid_dataset), 'valid', True)
    evaluate(writer, nli_net, optimizer, params, val_acc_best, stop_training, 0, test_dataloader, len(test_dataset), 'test', True)

    # Save encoder instead of full model
    torch.save(nli_net.encoder, os.path.join(params.outputdir, params.outputmodelname + '.encoder'))