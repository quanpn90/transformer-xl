import os, sys
import glob

from collections import Counter, OrderedDict
import numpy as np
import torch

from utils.vocabulary import Vocab


class LMOrderedIterator(object):
    def __init__(self, data, bsz, bptt, device='cpu', ext_len=None, bos_id=-1, eos_id=-1):
        """
            data -- LongTensor -- the LongTensor is strictly ordered
        """
        self.bsz = bsz
        self.bptt = bptt
        self.ext_len = ext_len if ext_len is not None else 0

        self.device = device

        # Work out how cleanly we can divide the dataset into bsz parts.
        self.n_step = data.size(0) // bsz

        # Trim off any extra elements that wouldn't cleanly fit (remainders).
        data = data.narrow(0, 0, self.n_step * bsz)

        weight = data.new(*data.size()).byte().fill_(1)

        if bos_id > 0:
            print("Creating target weights ...")
            weight.fill_(0)

            bos_ids = torch.nonzero(data.eq(bos_id)).squeeze().tolist()

            eos_ids = torch.nonzero(data.eq(eos_id)).squeeze().tolist()
            # print(bos_ids)
            if len(bos_ids) != len(eos_ids):
                print(len(eos_ids), len(bos_ids))
            assert(len(bos_ids) == len(eos_ids))

            # the weights inside these boundaries have value 1 and 0s elsewhere
            # still not optimized enough ...
            for (bos_, eos_) in zip(bos_ids, eos_ids):
                weight[bos_+1:eos_+1].fill_(1)

            print("Done")

        # Evenly divide the data across the bsz batches.
        self.data = data.view(bsz, -1).t().contiguous().to(device)
        self.weight = weight.view(bsz, -1).t().contiguous().to(device)

        # Number of mini-batches
        self.n_batch = (self.n_step + self.bptt - 1) // self.bptt

    def get_batch(self, i, bptt=None):
        if bptt is None: bptt = self.bptt
        seq_len = min(bptt, self.data.size(0) - 1 - i)

        end_idx = i + seq_len
        beg_idx = max(0, i - self.ext_len)

        data = self.data[beg_idx:end_idx]
        target = self.data[i+1:i+1+seq_len]
        weight = self.weight[i+1:i+1+seq_len]

        return data, target, seq_len, weight

    def get_fixlen_iter(self, start=0):
        for i in range(start, self.data.size(0) - 1, self.bptt):
            yield self.get_batch(i)

    # Variable length iteration
    def get_varlen_iter(self, start=0, std=5, min_len=5, max_deviation=3):
        max_len = self.bptt + max_deviation * std
        i = start
        while True:
            # 95% of the time long bptt, 5% half
            bptt = self.bptt if np.random.random() < 0.95 else self.bptt / 2.

            bptt = min(max_len, max(min_len, int(np.random.normal(bptt, std))))
            data, target, seq_len, weight = self.get_batch(i, bptt)
            i += seq_len
            yield data, target, seq_len, weight
            if i >= self.data.size(0) - 2:
                break

    def __iter__(self):
        return self.get_fixlen_iter()


class LMShuffledIterator(object):
    def __init__(self, data, bsz, bptt, device='cpu', ext_len=None, shuffle=False):
        """
            data -- list[LongTensor] -- there is no order among the LongTensors
        """
        self.data = data

        # first: sort the data by size
        sorted_data = sorted(data, key=lambda x: x.size(0))
        self.data = sorted_data

        self.bsz = bsz
        self.bptt = bptt
        self.ext_len = ext_len if ext_len is not None else 0

        self.device = device
        self.shuffle = shuffle

        self.batches = []
        self.multiplier = 8

        self.max_size = bsz * bptt  # maximum number of words in this minibatch

        # allocate the sentences into groups

        def _oversized(_cur_length, _n_sentences, _cur_batch_sizes, bsz_length, bsz_tokens):
            if n_sentences > bsz_length:
                return True

            if _n_sentences == 0:
                return False

            if (max(max(_cur_batch_sizes), max_size)) * (_n_sentences + 1) > bsz_tokens:
                return True

            return False

        # batch allocation
        cur_batch = []  # a list to store the sentence ids
        cur_batch_sizes = []
        i = 0
        while i < len(self.data):
            current_length = self.data[i].size(0)

            oversized = _oversized(current_length, len(cur_batch), _cur_batch_sizes, self.max_size, self.bsz)

            if oversized:
                # cut-off the current list to fit the multiplier
                current_size = len(cur_batch)
                scaled_size = max(
                    self.multiplier * (current_size // self.multiplier),
                    current_size % self.multiplier)
                batch_ = cur_batch[:scaled_size]
                self.batches.append(batch_)  # add this batch into the batch list

                cur_batch = cur_batch[scaled_size:]  # reset the current batch
                cur_batch_sizes = cur_batch_sizes[scaled_size:]

            cur_batch.append(i)
            cur_batch_sizes.append(current_length)

            i = i + 1

        # catch the last batch
        if len(cur_batch) > 0:
            self.batches.append(cur_batch)

        self.num_batches = len(self.batches)

    def collate(self, data):
        """
        :param data: list of sentences
        :return: a tensor
        """

        lengths = [x.size(0) for x in data]
        max_length = max(lengths)

        # tensor size: T x B
        tensor = data[0].new(max_length, len(data)).fill_(onmt.Constants.PAD)
        weights = tensor.new(*tensor.size()).zero_()

    # def get_sent_stream(self):
    #     # index iterator
    #     epoch_indices = np.random.permutation(len(self.data)) if self.shuffle \
    #         else np.array(range(len(self.data)))
    #
    #     # sentence iterator
    #     for idx in epoch_indices:
    #         yield self.data[idx]
    #
    # def stream_iterator(self, sent_stream):
    #     # streams for each data in the batch
    #     streams = [None] * self.bsz
    #
    #     data = torch.LongTensor(self.bptt, self.bsz)
    #     target = torch.LongTensor(self.bptt, self.bsz)
    #
    #     n_retain = 0
    #
    #     while True:
    #         # data   : [n_retain+bptt x bsz]
    #         # target : [bptt x bsz]
    #         data[n_retain:].fill_(-1)
    #         target.fill_(-1)
    #
    #         valid_batch = True
    #
    #         for i in range(self.bsz):
    #             n_filled = 0
    #             try:
    #                 while n_filled < self.bptt:
    #                     # get next sentence
    #                     if streams[i] is None or len(streams[i]) <= 1:
    #                         streams[i] = next(sent_stream)
    #                     # number of new tokens to fill in
    #                     n_new = min(len(streams[i]) - 1, self.bptt - n_filled)
    #                     # first n_retain tokens are retained from last batch
    #                     data[n_retain+n_filled:n_retain+n_filled+n_new, i] = \
    #                         streams[i][:n_new]
    #                     target[n_filled:n_filled+n_new, i] = \
    #                         streams[i][1:n_new+1]
    #                     streams[i] = streams[i][n_new:]
    #                     n_filled += n_new
    #             except StopIteration:
    #                 valid_batch = False
    #                 break
    #
    #         if not valid_batch:
    #             return
    #
    #         data = data.to(self.device)
    #         target = target.to(self.device)
    #
    #         yield data, target, self.bptt
    #
    #         n_retain = min(data.size(0), self.ext_len)
    #         if n_retain > 0:
    #             data[:n_retain] = data[-n_retain:]
    #         data.resize_(n_retain + self.bptt, data.size(1))

    def __iter__(self):
        # sent_stream is an iterator
        # sent_stream = self.get_sent_stream()

        # how do we get next batch ...

        for batch in self.stream_iterator(sent_stream):
            yield batch


class LMMultiFileIterator(LMShuffledIterator):
    def __init__(self, paths, vocab, bsz, bptt, device='cpu', ext_len=None,
        shuffle=False):

        self.paths = paths
        self.vocab = vocab

        self.bsz = bsz
        self.bptt = bptt
        self.ext_len = ext_len if ext_len is not None else 0

        self.device = device
        self.shuffle = shuffle

    def get_sent_stream(self, path):
        sents = self.vocab.encode_file(path, add_double_eos=True)
        if self.shuffle:
            np.random.shuffle(sents)
        sent_stream = iter(sents)

        return sent_stream

    def __iter__(self):
        if self.shuffle:
            np.random.shuffle(self.paths)

        for path in self.paths:
            # sent_stream is an iterator
            sent_stream = self.get_sent_stream(path)
            for batch in self.stream_iterator(sent_stream):
                yield batch


class Corpus(object):
    def __init__(self, path, dataset, order=True, *args, **kwargs):
        self.dataset = dataset
        self.vocab = Vocab(*args, **kwargs)
        self.order = True

        # if self.dataset in ['ptb', 'wt2', 'enwik8', 'text8', 'bilingual_ted']:
        #     self.vocab.count_file(os.path.join(path, 'train.txt'))
        #     self.vocab.count_file(os.path.join(path, 'valid.txt'))
        #     self.vocab.count_file(os.path.join(path, 'test.txt'))
        # elif self.dataset == 'wt103':
        #     self.vocab.count_file(os.path.join(path, 'train.txt'))
        # elif self.dataset == 'lm1b':
        #     train_path_pattern = os.path.join(
        #         path, '1-billion-word-language-modeling-benchmark-r13output',
        #         'training-monolingual.tokenized.shuffled', 'news.en-*')
        #     train_paths = glob.glob(train_path_pattern)
        #     # the vocab will load from file when build_vocab() is called

        self.vocab.count_file(os.path.join(path, 'train.txt'))
        self.vocab.build_vocab()

        self.train = self.vocab.encode_file(
            os.path.join(path, 'train.txt'), ordered=order)
        self.valid = self.vocab.encode_file(
            os.path.join(path, 'valid.txt'), ordered=order)
        self.test = self.vocab.encode_file(
            os.path.join(path, 'test.txt'), ordered=order)

        # if self.dataset in ['ptb', 'wt2', 'wt103']:
        #
        # elif self.dataset in ['enwik8', 'text8', 'bilingual_ted']:
        #     print("Creating %s dataset" % self.dataset)
        #     self.train = self.vocab.encode_file(
        #         os.path.join(path, 'train.txt'), ordered=True, add_eos=False)
        #     self.valid = self.vocab.encode_file(
        #         os.path.join(path, 'valid.txt'), ordered=True, add_eos=False)
        #     self.test  = self.vocab.encode_file(
        #         os.path.join(path, 'test.txt'), ordered=True, add_eos=False)
        # elif self.dataset == 'lm1b':
        #     self.train = train_paths
        #     self.valid = self.vocab.encode_file(
        #         os.path.join(path, 'valid.txt'), ordered=False, add_double_eos=True)
        #     self.test  = self.vocab.encode_file(
        #         os.path.join(path, 'test.txt'), ordered=False, add_double_eos=True)

    def get_iterator(self, split, *args, **kwargs):
        # if split == 'train':
        #     if self.dataset in ['ptb', 'wt2', 'wt103', 'enwik8', 'text8', 'bilingual_ted']:
        #         data_iter = LMOrderedIterator(self.train, *args, **kwargs)
        #     elif self.dataset == 'lm1b':
        #         kwargs['shuffle'] = True
        #         data_iter = LMMultiFileIterator(self.train, self.vocab, *args, **kwargs)
        # elif split in ['valid', 'test']:
        #     data = self.valid if split == 'valid' else self.test
        #     if self.dataset in ['ptb', 'wt2', 'wt103', 'enwik8', 'text8', 'bilingual_ted']:
        #         data_iter = LMOrderedIterator(data, *args, **kwargs)
        #     elif self.dataset == 'lm1b':
        #         data_iter = LMShuffledIterator(data, *args, **kwargs)

        if not hasattr(self, 'order'):
            self.order = True

        if self.order:
            if split == 'train':
                data_iter = LMOrderedIterator(self.train, *args, **kwargs)
            elif split in ['valid', 'test']:
                data_iter = LMOrderedIterator(self.valid, *args, **kwargs)
        else:
            if split == 'train':
                data_iter = LMShuffledIterator(self.train, *args, **kwargs)
            elif split in ['valid', 'test']:
                data_iter = LMShuffledIterator(self.valid, *args, **kwargs)

        return data_iter


def get_lm_corpus(datadir, dataset, order=True):
    fn = os.path.join(datadir, 'cache.pt')
    if os.path.exists(fn):
        print('Loading cached dataset...')
        corpus = torch.load(fn)
    else:
        print('Producing dataset {}...'.format(dataset))
        kwargs = {}
        kwargs['special'] = ["<pad>", '<unk>', '<bos>', '<eos>']
        kwargs['lower_case'] = False
        # if dataset in ['wt103', 'wt2']:
        #     kwargs['special'] = ['<eos>']
        #     kwargs['lower_case'] = False
        # elif dataset == 'ptb':
        #     kwargs['special'] = ['<eos>']
        #     kwargs['lower_case'] = True
        # elif dataset == 'bilingual_ted':
        #     kwargs['special'] = ['<eos>']
        #     kwargs['lower_case'] = False
        # elif dataset == 'lm1b':
        #     kwargs['special'] = []
        #     kwargs['lower_case'] = False
        #     kwargs['vocab_file'] = os.path.join(datadir, '1b_word_vocab.txt')
        # elif dataset in ['enwik8', 'text8']:
        #     pass

        corpus = Corpus(datadir, dataset, order=order, **kwargs)
        torch.save(corpus, fn)

    return corpus


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='unit test')
    parser.add_argument('--datadir', type=str, default='../data/text8',
                        help='location of the data corpus')
    parser.add_argument('--dataset', type=str, default='text8',
                        choices=['ptb', 'wt2', 'wt103', 'lm1b', 'enwik8', 'text8', 'bilingual_ted'],
                        help='dataset name')
    args = parser.parse_args()

    corpus = get_lm_corpus(args.datadir, args.dataset)
    print('Vocab size : {}'.format(len(corpus.vocab.idx2sym)))
