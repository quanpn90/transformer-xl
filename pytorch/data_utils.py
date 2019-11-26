import os, sys
import glob

from collections import Counter, OrderedDict
import numpy as np
import torch

from utils.vocabulary import Vocab


class LMOrderedIterator(object):
    def __init__(self, data, bsz, bptt, device='cpu', ext_len=None, bos_id=-1, eos_id=-1, **kwargs):
        """
            data -- LongTensor -- the LongTensor is strictly ordered
        """
        self.bsz = 1
        self.bptt = bptt
        self.ext_len = ext_len if ext_len is not None else 0

        self.device = device
        self.data = data  # don't sort the data

        # batch allocation
        self.batches = []
        cur_batch = []  # a list to store the sentence ids
        cur_size = 0
        self.bos_id = bos_id
        self.eos_id = eos_id
        i = 0

        def _oversized(_cur_length, _cur_batch_size, max_length):

            if _cur_batch_size + _cur_length > max_length:
                return True

            return False

        while i < len(self.data):

            current_length = self.data[i].size(0)
            #
            oversized = _oversized(current_length, cur_size, self.bptt)
            #
            if oversized:

                self.batches.append(cur_batch)
                # reset the batch
                cur_batch = []
                cur_size = 0

            cur_batch.append(i)
            cur_size += current_length
            i = i+1

        # catch the last batch
        if len(cur_batch) > 0:
            self.batches.append(cur_batch)

        self.num_batches = len(self.batches)
        self.order = torch.arange(self.num_batches)
        self.cur_index = 0

    def get_batch(self, i):
        """
        :param i: the index of the mini batch
        :return: data_input, data_target, data_length, data_weight
        """

        sent_ids = self.batches[i]
        data = [self.data[i] for i in sent_ids]
        lengths = [x.size(0) for x in data]
        max_length = sum(lengths)

        # tensor size: T x 1
        # tensor = data[0].new(max_length, bsz).fill_(0)
        tensor = data[0].new(1, max_length)
        weight = tensor.new(*tensor.size()).fill_(0)

        # start from position 0
        offset = 0

        for i in range(len(data)):
            data_length = data[i].size(0)
            tensor[0].narrow(0, offset, data_length).copy_(data[i])

            if self.bos_id > 0:
                bos_pos = torch.nonzero(data[i].eq(self.bos_id)).squeeze().item()
                eos_pos = torch.nonzero(data[i].eq(self.eos_id)).squeeze().item()
                length = eos_pos - bos_pos + 1
                weight[0].narrow(0, offset + bos_pos, length).fill_(1)
            else:
                weight[0].narrow(0, offset, data_length).fill_(1)

            # move the offset to the next sentence
            offset = offset + data_length

        tensor = tensor.transpose(0, 1).contiguous().to(self.device)
        weight = weight.transpose(0, 1).contiguous().to(self.device)
        input_ = tensor[:-1, :]
        target = tensor[1:, :]
        weight = weight[1:, :]

        return input_, target, max_length, weight

    #     # Trim off any extra elements that wouldn't cleanly fit (remainders).
    #     data = data.narrow(0, 0, self.n_step * bsz)
    #
    #     weight = data.new(*data.size()).byte().fill_(1)
    #
    #     if bos_id > 0:
    #         print("Creating target weights ...")
    #         weight.fill_(0)
    #
    #         bos_ids = torch.nonzero(data.eq(bos_id)).squeeze().tolist()
    #
    #         eos_ids = torch.nonzero(data.eq(eos_id)).squeeze().tolist()
    #         # print(bos_ids)
    #         if len(bos_ids) != len(eos_ids):
    #             print(len(eos_ids), len(bos_ids))
    #         assert(len(bos_ids) == len(eos_ids))
    #
    #         # the weights inside these boundaries have value 1 and 0s elsewhere
    #         # still not optimized enough ...
    #         for (bos_, eos_) in zip(bos_ids, eos_ids):
    #             weight[bos_+1:eos_+1].fill_(1)
    #
    #         print("Done")
    #
    #     # Evenly divide the data across the bsz batches.
    #     self.data = data.view(bsz, -1).t().contiguous().to(device)
    #     self.weight = weight.view(bsz, -1).t().contiguous().to(device)
    #
    #     # Number of mini-batches
    #     self.n_batch = (self.n_step + self.bptt - 1) // self.bptt
    #
    # def get_batch(self, i, bptt=None):
    #     if bptt is None: bptt = self.bptt
    #     seq_len = min(bptt, self.data.size(0) - 1 - i)
    #
    #     end_idx = i + seq_len
    #     beg_idx = max(0, i - self.ext_len)
    #
    #     data = self.data[beg_idx:end_idx]
    #     target = self.data[i+1:i+1+seq_len]
    #     weight = self.weight[i+1:i+1+seq_len]
    #
    #     return data, target, seq_len, weight
    #
    def get_fixlen_iter(self, start=0):
        for i in range(start, self.data.size(0) - 1, self.bptt):
            yield self.get_batch(i)

    def next(self):
        if self.cur_index >= self.num_batches:
            self.cur_index = 0
            self.reset_order()

        batch = self.get_batch(self.cur_index)
        self.cur_index = self.cur_index + 1

        return batch
    #
    # # Variable length iteration
    # def get_varlen_iter(self, start=0, std=5, min_len=5, max_deviation=3):
    #     max_len = self.bptt + max_deviation * std
    #     i = start
    #     while True:
    #         # 95% of the time long bptt, 5% half
    #         bptt = self.bptt if np.random.random() < 0.95 else self.bptt / 2.
    #
    #         bptt = min(max_len, max(min_len, int(np.random.normal(bptt, std))))
    #         data, target, seq_len, weight = self.get_batch(i, bptt)
    #         i += seq_len
    #         yield data, target, seq_len, weight
    #         if i >= self.data.size(0) - 2:
    #             break

    def __iter__(self):
        # how do we get next batch ...
        for i in range(self.num_batches):
            batch = self.get_batch(i)
            yield batch

    def reset_order(self):
        return


class LMShuffledIterator(object):
    def __init__(self, data, bsz, bptt, device='cpu', ext_len=None, bos_id=-1, eos_id=-1, **kwargs):
        """
            data -- list[LongTensor] -- there is no order among the LongTensors
        """
        self.data = data

        # first: sort the data by size
        sorted_data = sorted(data, key=lambda x: x.size(0))
        self.data = sorted_data
        self.bos_id = bos_id
        self.eos_id = eos_id

        self.bsz = bsz
        self.bptt = bptt
        self.ext_len = ext_len if ext_len is not None else 0

        self.device = device

        self.batches = []
        self.multiplier = 8

        self.max_size = bptt  # maximum number of words in this minibatch

        # allocate the sentences into groups
        def _oversized(_cur_length, _n_sentences, _cur_batch_sizes, bsz_length, bsz_tokens):
            if _n_sentences > bsz_length:
                return True

            if _n_sentences == 0:
                return False

            max_size = max(_cur_batch_sizes)
            if (max(max(_cur_batch_sizes), max_size)) * (_n_sentences + 1) > bsz_tokens:
                return True

            return False

        # batch allocation
        cur_batch = []  # a list to store the sentence ids
        cur_batch_sizes = []
        i = 0
        while i < len(self.data):
            current_length = self.data[i].size(0)

            oversized = _oversized(current_length, len(cur_batch), cur_batch_sizes, self.bsz ,self.max_size)

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
        self.order = torch.randperm(self.num_batches)
        self.cur_index = 0

    def reset_order(self):
        self.order = torch.randperm(self.num_batches)

    def next(self):
        if self.cur_index >= self.num_batches:
            self.cur_index = 0
            self.reset_order()

        batch = self.get_batch(self.order[self.cur_index])
        self.cur_index = self.cur_index + 1

        return batch

    def get_batch(self, i):
        """
        :param data: list of sentences
        :return: a tensor
        """

        sent_ids = self.batches[i]
        data = [self.data[i] for i in sent_ids]
        lengths = [x.size(0) for x in data]
        max_length = max(lengths)

        # tensor size: T x B
        # tensor = data[0].new(max_length, len(data)).fill_(0)
        tensor = data[0].new(len(data), max_length).fill_(0)
        weight = tensor.new(*tensor.size()).fill_(0)

        for i in range(len(data)):
            data_length = data[i].size(0)
            offset = 0  # align to the left
            tensor[i].narrow(0, offset, data_length).copy_(data[i])

            if self.bos_id > 0:
                bos_pos = torch.nonzero(data[i].eq(self.bos_id)).squeeze().item()
                eos_pos = torch.nonzero(data[i].eq(self.eos_id)).squeeze().item()
                length = eos_pos - bos_pos + 1
                weight[i].narrow(0, bos_pos, length).fill_(1)
            else:
                weight[:, i].narrow(0, offset, data_length).fill_(1)

        tensor = tensor.transpose(0, 1).contiguous().to(self.device)
        weight = weight.transpose(0, 1).contiguous().to(self.device)
        input_ = tensor[:-1, :]
        target = tensor[1:, :]
        weight = weight[1:, :]

        return input_, target, max_length, weight

    # def __iter__(self):
    #     # sent_stream is an iterator
    #     # sent_stream = self.get_sent_stream()
    #
    #     # how do we get next batch ...
    #
    #     for batch in self.stream_iterator(sent_stream):
    #         yield batch

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
        # how do we get next batch ...
        for i in range(self.num_batches):
            batch = self.get_batch(self.order[i])
            yield batch


class Corpus(object):
    def __init__(self, path, dataset, *args, **kwargs):
        self.dataset = dataset
        self.vocab = Vocab(*args, **kwargs)
        # self.order = kwargs.get('order', True)

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
            os.path.join(path, 'train.txt'))
        self.valid = self.vocab.encode_file(
            os.path.join(path, 'valid.txt'))
        self.test = self.vocab.encode_file(
            os.path.join(path, 'test.txt'))

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

        # if not hasattr(self, 'order'):
        #     self.order = True
        order = kwargs.get('order', True)

        if order:
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


def get_lm_corpus(datadir, dataset):
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

        corpus = Corpus(datadir, dataset, **kwargs)
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
