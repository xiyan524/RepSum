import os
import codecs
import json
import numpy as np
from abc import ABC

from nltk import word_tokenize
from tabulate import tabulate
from torch.utils.data import Dataset

from modules.data.utils import vectorize, read_corpus, read_corpus_subw, \
 unks_per_sample, token_swaps, read_corpus_dialogue, unks_per_sample_dialogue, shuffle

class BaseLMDataset(Dataset, ABC):
    def __init__(self, input_path, mode, k, batch_size=None, summary_path=None, nsent_path=None, org_dia_path=None, preprocess=None,
                 vocab=None, vocab_size=None, subword=False, subword_path=None, verbose=True, **kwargs):
        """
        Base Dataset for Language Modeling.

        Args:
            preprocess (callable): preprocessing callable, which takes as input
                a string and returns a list of tokens
            #input (str, list): the path to the data file, or a list of samples.
            input_path: the path to the data file
            vocab (Vocab): a vocab instance. If None, then build a new one
                from the Datasets data.
            vocab_size(int): if given, then trim the vocab to the given number.
            subword(bool): whether the dataset will be
                tokenized using subword units, using the SentencePiece package.
            subword(SentencePieceProcessor): path to the sentencepiece model
            verbose(bool): print useful statistics about the dataset.
        """
        self.mode = mode
        self.batch_size = batch_size
        self.k = k
        if mode != "test":
            self.inputs, self.inputs_sim, self.decoder_inputs = load_data_cl_sec(input_path)
            #self.inputs, self.inputs_sim, self.decoder_inputs = load_data_cl_ami(input_path)
        else:
            self.inputs, self.inputs_sim, self.decoder_inputs = load_data_cl_test(input_path, batch_size, summary_path, nsent_path, org_dia_path)
            #self.inputs, self.inputs_sim, self.decoder_inputs = load_data_cl_test_ami(input_path, batch_size, summary_path, nsent_path, org_dia_path)
        self.subword = subword
        self.subword_path = subword_path

        if preprocess is not None:
            self.preprocess = preprocess

        # tokenize the dataset
        if self.subword:
            self.vocab, self.data = read_corpus_subw(self.inputs, subword_path)
        else:
            #self.vocab, self.data = read_corpus(self.inputs, self.preprocess)
            _, self.decoder_inputs = read_corpus(self.decoder_inputs, self.preprocess)
            self.vocab, self.data = read_corpus_dialogue(self.inputs, self.preprocess)
            _, self.data_sim = read_corpus_dialogue(self.inputs_sim, self.preprocess)

        if vocab is not None:
            self.vocab = vocab
        else:
            self.vocab.build(vocab_size)

        if verbose:
            print(self)
            print()

    def __str__(self):

        props = []
        """if isinstance(self.input, str):
            props.append(("source", os.path.basename(self.input)))"""
        _covarage = unks_per_sample_dialogue(self.vocab.tok2id.keys(), self.data)
        _covarage = str(_covarage.round(4)) + " %"

        try:
            props.append(("size", len(self)))
        except:
            pass
        props.append(("vocab size", len(self.vocab)))
        props.append(("unique tokens", len(self.vocab.vocab)))
        props.append(("UNK per sample", _covarage))
        props.append(("subword", self.subword))

        if hasattr(self, 'seq_len'):
            props.append(("max seq length", self.seq_len))
        if hasattr(self, 'bptt'):
            props.append(("BPTT", self.bptt))
        if hasattr(self, 'attributes'):
            props.append(("attributes", len(self.attributes[0])))

        return tabulate([[x[1] for x in props]], headers=[x[0] for x in props])

    def truncate(self, n):
        self.data = self.data[:n]

    @staticmethod
    def preprocess(text, lower=True):
        if lower:
            text = text.lower()
        # return text.split()
        return word_tokenize(text)


class DsDataset(BaseLMDataset):
    def __init__(self, *args, sent_num, sent_len, sent_sim_len, dec_seq_len, **kwargs):
        """Dataset for sequence dialogue summarization."""

        super().__init__(*args, **kwargs)
        # todo: find more elegant way to ignore seq_len
        self.sent_num = sent_num
        self.sent_len = sent_len
        self.sent_sim_len = sent_sim_len
        self.dec_seq_len = dec_seq_len
        self.oovs = kwargs.get("oovs", 0)
        self.return_oov = kwargs.get("return_oov", False)
        self.swaps = kwargs.get("swaps", 0.0)

        for i in range(self.oovs):
            self.vocab.add_token(f"<oov-{i}>")

    def __len__(self):
        return len(self.data)

    def read_sample(self, index):
        """calculate the idf for each dialogue"""
        sample = self.data[index][:self.seq_len]
        sample = [self.vocab.SOS] + sample + [self.vocab.EOS]
        sample, _ = vectorize(sample, self.vocab, self.oovs)
        return list(map(self.vocab.id2tok.get, sample))

    def __getitem__(self, index):
        inp_x = self.data[index][:self.sent_num]
        inp_sim = self.data_sim[index]
        out_sim = [0] * self.k
        inp_y = [self.vocab.SOS] + self.decoder_inputs[index][:self.dec_seq_len]
        out_y = self.decoder_inputs[index][:self.dec_seq_len] + [self.vocab.EOS]

        if not self.subword:
            imp_x_vec= []
            inp_sim_vec = []
            dialogue_len = []
            sim_len = []

            for sent in inp_x:
                inp_x_tmp, _ = vectorize(sent, self.vocab, self.oovs)
                imp_x_vec.append(inp_x_tmp)
                if len(sent) < self.sent_len:
                    dialogue_len.append(len(sent))
                else:
                    dialogue_len.append(self.sent_len)
            inp_x_oov_map = [x for j in inp_x for x in j]
            _, oov_map = vectorize(inp_x_oov_map, self.vocab, self.oovs)

            inp_sim.append(self.decoder_inputs[index])
            out_sim.append(1)
            inp_sim, out_sim = shuffle(inp_sim, out_sim)
            for sent in inp_sim:
                inp_sim_tmp, _ = vectorize(sent, self.vocab, self.oovs)
                inp_sim_vec.append(inp_sim_tmp)
                if len(sent) < self.sent_sim_len:
                    sim_len.append(len(sent))
                else:
                    sim_len.append(self.sent_sim_len)

            inp_x = imp_x_vec
            inp_sim = inp_sim_vec
            inp_y, _ = vectorize(inp_y, self.vocab, self.oovs)
            out_y, _ = vectorize(out_y, self.vocab, self.oovs)
        else:
            raise NotImplementedError

        if len(inp_x) < self.sent_num:
            sample = inp_x, inp_sim, out_sim, inp_y, out_y, sim_len, dialogue_len, len(inp_x), len(inp_y)
        else:
            sample = inp_x, inp_sim, out_sim, inp_y, out_y, sim_len, dialogue_len, self.sent_num, len(inp_y)
        
        if self.return_oov:
            sample = sample + (oov_map,)

        return sample



def load_data_cl_sec(path):
    # encoder
    inputs = []
    inputs_sim = []
    decoder_inputs = []

    file_names = os.listdir(path)
    for file_name in file_names:
        with open(os.path.join(path, file_name), 'r') as load_f:
            try:
                file_dict = json.load(load_f)
            except Exception:
                print(file_name)
                continue
            sec_num = file_dict["section_num"]
            for i in range(sec_num):
                sec_tmp = file_dict["section"+str(i)]
                sec_dialogue = sec_tmp["dialogue"]
                sec_n_sent = sec_tmp["n_sent"]
                sec_sim = sec_tmp["n_sent_sim"][:6]

                if len(sec_dialogue) < 10:
                    continue

                inputs.append(sec_dialogue)
                inputs_sim.append(sec_sim)
                decoder_inputs.append(sec_n_sent)
            sec_dialogue = file_dict["dialogue"]
            sec_n_sent = file_dict["n_sent"]
            sec_sim = file_dict["n_sent_sim"][:3]

            if len(sec_dialogue) < 10:
                continue

            inputs.append(sec_dialogue)
            inputs_sim.append(sec_sim)
            decoder_inputs.append(sec_n_sent)

    return inputs, inputs_sim, decoder_inputs


def load_data_cl_test(path, batch_size, summary_path, nsent_path, org_dia_path):
    inputs = []
    inputs_sim = []
    decoder_inputs = []
    summaries = []

    file_names = os.listdir(path)
    for file_name in file_names:
        with open(os.path.join(path, file_name), 'r') as load_f:
            try:
                file_dict = json.load(load_f)         
                input_dialogue = file_dict['unsupervised_dialogue']
                input_sim = file_dict["n_sent_sim"][:3]
                decoder_sentence = file_dict['n_sent']
                summary = file_dict['summary']
            except Exception:
                print(file_name)
                continue

            if len(input_dialogue) < 10:
                continue

        inputs.append(input_dialogue)
        inputs_sim.append(input_sim)
        decoder_inputs.append(decoder_sentence)
        summaries.append(summary)
    
    summary_file = codecs.open(summary_path, 'w')
    abandon_sample = len(inputs) % batch_size
    for sumf in summaries[:-abandon_sample]:
        summary_file.write(sumf+"\n")

    nsent_file = codecs.open(nsent_path, 'w')
    for nsent_tmp in decoder_inputs[:-abandon_sample]:
        nsent_file.write(nsent_tmp+"\n")

    return inputs, inputs_sim, decoder_inputs



