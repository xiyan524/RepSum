import torch
from torch.nn.utils.rnn import pad_sequence


class SeqCollate:
    """
    Base Class.
    A variant of callate_fn that pads according to the longest sequence in
    a batch of sequences
    """

    def __init__(self, sim_len, sent_num, sent_len, sort=False, batch_first=True):
        self.sort = sort
        self.batch_first = batch_first
        self.sim_len = sim_len
        self.sent_num = sent_num
        self.sent_len = sent_len

    def pad_samples(self, samples):
        return pad_sequence([torch.LongTensor(x) for x in samples], self.batch_first)

    def _collate(self, *args):
        raise NotImplementedError

    def __call__(self, batch):
        batch = list(zip(*batch))
        return self._collate(*batch)

    def pad_dialogues(self, samples):
        pad_lst = [0] * self.sent_len
        results_lst = []
        for sample in samples:
            for index in range(len(sample)):
                if len(sample[index]) >= self.sent_len:
                    sample[index] = sample[index][:self.sent_len]
                else:
                    while len(sample[index]) < self.sent_len:
                        sample[index].append(0)
            if len(sample) >= self.sent_num:
                sample = sample[:self.sent_num]
            else:
                while len(sample) < self.sent_num:
                    sample.append(pad_lst)
            
            #results.append(torch.LongTensor([i for item in sample for i in item]))
            results_lst.append([i for item in sample for i in item])
            results = torch.LongTensor(results_lst)
        return results

    def pad_sim_sent(self, samples):
        results_lst = []
        for sample in samples:
            for index in range(len(sample)):
                if len(sample[index]) >= self.sim_len:
                    sample[index] = sample[index][:self.sim_len]
                else:
                    while len(sample[index]) < self.sim_len:
                        sample[index].append(0)
            results_lst.append([i for item in sample for i in item])
            results = torch.LongTensor(results_lst)
        return results


class LMCollate(SeqCollate):
    def __init__(self, *args):
        super().__init__(*args)

    def _collate(self, inputs, targets, lengths):
        inputs = self.pad_samples(inputs)
        targets = self.pad_samples(targets)
        lengths = torch.LongTensor(lengths)
        return inputs, targets, lengths


class CondLMCollate(SeqCollate):
    def __init__(self, *args):
        super().__init__(*args)

    def _collate(self, inputs, targets, attributes, lengths):
        inputs = self.pad_samples(inputs)
        targets = self.pad_samples(targets)
        attributes = self.pad_samples(attributes)
        lengths = torch.LongTensor(lengths)
        return inputs, targets, attributes, lengths


class Seq2SeqCollate(SeqCollate):
    def __init__(self, *args):
        super().__init__(*args)
    
    def _collate(self, inp_src, inp_sim, out_sim, inp_trg, out_trg, sim_len, sent_len, sent_num, len_trg):
        inp_src = self.pad_dialogues(inp_src)
        inp_sim = self.pad_sim_sent(inp_sim)
        inp_trg = self.pad_samples(inp_trg)
        out_trg = self.pad_samples(out_trg)

        for index in range(len(sent_len)):
            while len(sent_len[index]) < self.sent_num:
                sent_len[index].append(5)

        out_sim = torch.LongTensor(out_sim)
        sim_len = torch.LongTensor(sim_len)
        sent_len = torch.LongTensor(sent_len)
        sent_num = torch.LongTensor(sent_num)
        len_trg = torch.LongTensor(len_trg)

        return inp_src, inp_sim, out_sim, inp_trg, out_trg, sim_len, sent_len, sent_num, len_trg


class Seq2SeqOOVCollate(SeqCollate):
    def __init__(self, *args):
        super().__init__(*args)

    def _collate(self, inp_src, inp_sim, out_sim, inp_trg, out_trg, sim_len, sent_len, sent_num, len_trg, oov_map):
        inp_src = self.pad_dialogues(inp_src)
        inp_sim = self.pad_sim_sent(inp_sim)
        inp_trg = self.pad_samples(inp_trg)
        out_trg = self.pad_samples(out_trg)

        for index in range(len(sent_len)):
            while len(sent_len[index]) < self.sent_num:
                sent_len[index].append(10)
        inp_sim = torch.LongTensor(inp_sim)
        
        out_sim = torch.LongTensor(out_sim)
        sim_len = torch.LongTensor(sim_len)
        sent_len = torch.LongTensor(sent_len)
        sent_num = torch.LongTensor(sent_num)
        len_trg = torch.LongTensor(len_trg)

        return inp_src, inp_sim, out_sim, inp_trg, out_trg, sim_len, sent_len, sent_num, len_trg, oov_map
