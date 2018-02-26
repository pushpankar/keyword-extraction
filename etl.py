from nltk.tokenize import word_tokenize
from functools import partial
from dataset import JobDataset
import torch
from torch.autograd import Variable
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from collections import Counter

def make_vocab(texts, vocab_size, add_end=True):
    words = [word for sentence in texts for word in word_tokenize(sentence)]
    common_words = [item for item, c in Counter(words).most_common(vocab_size)]
    rev_dict = dict(enumerate(common_words))
    if vocab_size:
        rev_dict[len(rev_dict)] = 'UNK'
    if add_end:
        rev_dict[len(rev_dict)] = "<<END>>"
    word_dict = {y:x for x, y in rev_dict.items()}
    return word_dict, rev_dict


def make_sent2num(vocab, seq_len, add_end=True):
    def _sent2nums(sent):
        words = word_tokenize(str(sent).lower())
        nums = list(map(lambda x: vocab[x]
                        if x in vocab
                        else vocab['UNK'],
                        words))
        if add_end:
            nums = (nums * seq_len)[:seq_len-1] + [vocab["<<END>>"]]
        else:
            nums = (nums * seq_len)[:seq_len]
        return nums
    return _sent2nums


def batch2nums(sentences, sent2num_func):
    batch_data = list(map(sent2num_func, sentences))
    batch_data = Variable(torch.LongTensor(batch_data))
    return batch_data


def get_loaders(df, batch_size, ratio=0.1, sort=False, split=True):
    train_data, test_data = train_test_split(df, test_size=ratio)
    train_data = train_data.reset_index()
    test_data = test_data.reset_index()
    if sort:
        train_data = sort_data(train_data, 'sentence')

    train_dataset = JobDataset(train_data)
    test_dataset = JobDataset(test_data)

    train_loader = DataLoader(train_dataset, batch_size,
                              drop_last=True)
    test_loader = DataLoader(test_dataset, batch_size,
                             drop_last=True)
    return train_loader, test_loader


def max_len(data):
    return max(map(lambda x: len(word_tokenize(x)), data))


def class_weights(classes):
    counts = classes.value_counts()
    weights = [sum(counts) / count for count in counts]
    return weights


def sort_data(df, by_col):
    fun = lambda x: len(word_tokenize(x))
    df = df.assign(f = df[by_col].map(fun))
    sorted_df = df.sort_values('f')
    df = sorted_df.drop('f', axis=1)
    return df
