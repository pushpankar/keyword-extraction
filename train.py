from net import Net
import pandas as pd
from etl import get_loaders, make_vocab, batch2nums, make_sent2num

import torch
from torch.autograd import Variable
import argparse
import torch.nn as nn
from torch.optim import Adam
from tensorboard_logger import log_value, configure


def load_data():
    df = pd.read_csv("./data.tsv", sep="\t", header=None)
    return df


def main(args):
    net = Net(args.vocab_size, args.hidden_size, args.embed_size)
    data = load_data()
    
    critic = nn.CosineEmbeddingLoss()
    params = filter(lambda p: p.requires_grad, net.parameters())
    optimizer = Adam(params, lr=args.lr)

    max_words = args.vocab_size

    vocab, _ = make_vocab(data[2], max_words)
    
    train_loader, test_loader = get_loaders(data, args.batch_size, 0.1, False)
    sent2num_func = make_sent2num(vocab, args.seq_len)

    step = 0
    for e in range(args.max_epochs):
        for i, batch_data in enumerate(train_loader):
            step += 1
            optimizer.zero_grad()
            description_tensor = batch2nums(batch_data["description"], sent2num_func)
            title_tensor = batch2nums(batch_data["title"], sent2num_func)
            y = Variable(torch.LongTensor(batch_data["related"]))
            context_vec, title_vec, att = net(description_tensor, title_tensor)
            loss = critic(context_vec, title_vec, y)
            loss.backward()
            optimizer.step()

            log_value("loss", loss.data[0], step=step)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Inputs")
    parser.add_argument("--vocab_size", type=int, dest="vocab_size",
                        default=20000)
    parser.add_argument("--embed_size", type=int, dest="embed_size",
                        default=200)
    parser.add_argument("--hidden_size", type=int, dest="hidden_size",
                        default=300)
    parser.add_argument("--batch_size", type=int, dest="batch_size",
                        default=32)
    parser.add_argument("--lr", type=float, dest="lr",
                        default=1e-4)
    parser.add_argument("--max_epochs", type=int, dest="max_epochs",
                        default=100)
    parser.add_argument("--seq_len", type=int, dest="seq_len",
                        default=30)

    args = parser.parse_args()
    configure("logs/init")
    main(args)
