import torch
from torch import nn
from torch.nn import functional as F

from modules.modules import RecurrentHelper, AttSeqDecoder, SeqReader
from modules.helpers import sequence_mask, avg_vectors, index_mask


class Seq2Seq2Seq(nn.Module, RecurrentHelper):

    def __init__(self, n_tokens, **kwargs):
        super(Seq2Seq2Seq, self).__init__()

        ############################################
        # Attributes
        ############################################
        self.n_tokens = n_tokens
        self.bridge_hidden = kwargs.get("bridge_hidden", False)
        self.bridge_non_linearity = kwargs.get("bridge_non_linearity", None)
        self.detach_hidden = kwargs.get("detach_hidden", False)
        self.input_feeding = kwargs.get("input_feeding", False)
        self.length_control = kwargs.get("length_control", False)
        self.bi_encoder = kwargs.get("rnn_bidirectional", False)
        self.rnn_type = kwargs.get("rnn_type", "LSTM")
        self.layer_norm = kwargs.get("layer_norm", False)
        self.sos = kwargs.get("sos", 1)
        self.sample_embed_noise = kwargs.get("sample_embed_noise", 0)
        self.topic_idf = kwargs.get("topic_idf", False)
        self.dec_token_dropout = kwargs.get("dec_token_dropout", .0)
        self.enc_token_dropout = kwargs.get("enc_token_dropout", .0)

        self.batch_size = kwargs.get("batch_size")
        self.sent_num = kwargs.get("sent_num")
        self.sent_len = kwargs.get("sent_len")

        # tie embedding layers to output layers (vocabulary projections)
        kwargs["tie_weights"] = kwargs.get("tie_embedding_outputs", False)

        ############################################
        # Layers
        ############################################

        # backward-compatibility for older version of the project
        kwargs["rnn_size"] = kwargs.get("enc_rnn_size", kwargs.get("rnn_size"))
        self.inp_encoder = SeqReader(self.n_tokens, **kwargs)
        enc_size = self.inp_encoder.rnn_size
        self.sent_classification = torch.nn.Linear(enc_size, 1)
        self.sent_similar = torch.nn.Linear(enc_size*2, 1)

        # backward-compatibility for older version of the project
        kwargs["rnn_size"] = kwargs.get("dec_rnn_size", kwargs.get("rnn_size"))   
        self.dia_nsent = AttSeqDecoder(self.n_tokens, enc_size, **kwargs)
        self.sum_nsent = AttSeqDecoder(self.n_tokens, enc_size, **kwargs)

        # create a dummy embedding layer, which will retrieve the idf values
        # of each word, given the word ids
        if self.topic_idf:
            self.idf = nn.Embedding(num_embeddings=n_tokens, embedding_dim=1)
            self.idf.weight.requires_grad = False

        if self.bridge_hidden:
            self._initialize_bridge(enc_size,
                                    kwargs["dec_rnn_size"],
                                    kwargs["rnn_layers"])

    def _initialize_bridge(self, enc_hidden_size, dec_hidden_size, num_layers):
        """
        adapted from
        https://github.com/OpenNMT/OpenNMT-py/blob/master/onmt/encoders/rnn_encoder.py#L85
        """

        # LSTM has hidden and cell state, other only one
        number_of_states = 2 if self.rnn_type == "LSTM" else 1

        if self.length_control:
            # add a parameter, for scaling the absolute target length
            self.Wl = nn.Parameter(torch.rand(1))
            # the length information will contain 2 additional dimensions,
            # - the target length
            # - the expansion / compression ratio given the source length
            enc_hidden_size += 2

        # Build a linear layer for each
        self.src_bridge = nn.ModuleList([nn.Linear(enc_hidden_size,
                                                   dec_hidden_size)
                                         for _ in range(number_of_states)])
        self.trg_bridge = nn.ModuleList([nn.Linear(enc_hidden_size,
                                                   dec_hidden_size)
                                         for _ in range(number_of_states)])

    def _bridge(self, bridge, hidden, src_lengths=None, trg_lengths=None):
        """Forward hidden state through bridge."""

        def _fix_hidden(_hidden):
            # The encoder hidden is  (layers*directions) x batch x dim.
            # We need to convert it to layers x batch x (directions*dim).
            fwd_final = _hidden[0:_hidden.size(0):2]
            bwd_final = _hidden[1:_hidden.size(0):2]
            final = torch.cat([fwd_final, bwd_final], dim=2)
            return final

        def bottle_hidden(linear, states, length_feats=None):
            if length_feats is not None:
                lf = length_feats.unsqueeze(0).repeat(states.size(0), 1, 1)
                _states = torch.cat([states, lf], -1)
                result = linear(_states)
            else:
                result = linear(states)

            if self.bridge_non_linearity == "tanh":
                result = torch.tanh(result)
            elif self.bridge_non_linearity == "relu":
                result = F.relu(result)

            return result

        if self.length_control:
            ratio = trg_lengths.float() / src_lengths.float()
            lengths = trg_lengths.float() * self.Wl
            L = torch.stack([ratio, lengths], -1)
        else:
            L = None

        if isinstance(hidden, tuple):  # LSTM
            # concat directions
            hidden = tuple(_fix_hidden(h) for h in hidden)
            outs = tuple([bottle_hidden(state, hidden[ix], L)
                          for ix, state in enumerate(bridge)])
        else:
            outs = bottle_hidden(bridge[0], hidden)

        return outs

    def initialize_embeddings(self, embs, trainable=False):

        freeze = not trainable

        embeddings = torch.from_numpy(embs).float()
        embedding_layer = nn.Embedding.from_pretrained(embeddings, freeze)

        self.inp_encoder.embed.embedding = embedding_layer
        self.cmp_encoder.embed.embedding = embedding_layer
        self.compressor.embed.embedding = embedding_layer
        self.decompressor.embed.embedding = embedding_layer
        self.original_task.embed.embedding = embedding_layer

    def initialize_embeddings_idf(self, idf):
        idf_embs = torch.from_numpy(idf).float().unsqueeze(-1)
        self.idf = nn.Embedding.from_pretrained(idf_embs, freeze=True)

    def set_embedding_gradient_mask(self, mask):
        self.inp_encoder.embed.set_grad_mask(mask)
        self.cmp_encoder.embed.set_grad_mask(mask)
        self.compressor.embed.set_grad_mask(mask)
        self.decompressor.embed.set_grad_mask(mask)
        self.original_task.embed.set_grad_mask(mask)

    def _fake_inputs(self, inputs, latent_lengths, pad=1):
        batch_size, seq_len = inputs.size()

        if latent_lengths is not None:
            max_length = max(latent_lengths)
        else:
            max_length = seq_len + pad

        fakes = torch.zeros(batch_size, max_length, device=inputs.device)
        fakes = fakes.type_as(inputs)
        fakes[:, 0] = self.sos
        return fakes

    #def generate(self, inputs, nsent, src_lengths, trg_seq_len, nsent_len, sampling):
    def generate(self, inputs, src_lengths, trg_seq_len):
        # ENCODER1
        enc1_results = self.inp_encoder(inputs, None, src_lengths)
        outs_enc1, hn_enc1 = enc1_results[-2:]

        # DECODER1
        dec_init = self._bridge(self.src_bridge, hn_enc1, src_lengths, trg_seq_len)
        inp_fake = self._fake_inputs(inputs, trg_seq_len)
        dec1_results = self.compressor(inp_fake, outs_enc1, dec_init,
                                       argmax=True,
                                       enc_lengths=src_lengths,
                                       sampling_prob=1.,
                                       desired_lengths=trg_seq_len)

        return enc1_results, dec1_results

    def summary(self, inp_src, sent_len, sent_num):
        imp_src_org = inp_src.view(self.batch_size, self.sent_num, self.sent_len)
        inp_src = imp_src_org.view(imp_src_org.size(0) * imp_src_org.size(1), imp_src_org.size(2))
        inp_length = imp_src_org.size(2)
        sent_len = sent_len.view(self.batch_size * self.sent_num)
        enc1_results = self.inp_encoder(inp_src, None, sent_len, word_dropout=self.enc_token_dropout)
        outs_enc, hn_enc = enc1_results[-2:]

        sent_len_mask = torch.unsqueeze(sequence_mask(sent_len, max_len=self.sent_len), -1).float()
        outs_enc = torch.mul(outs_enc, sent_len_mask)
        outs_enc = outs_enc.view(self.batch_size, self.sent_num, self.sent_len, -1)
        outs_enc = torch.sum(outs_enc, dim=2)

        sent_num_mask = torch.unsqueeze(sequence_mask(sent_num, max_len=self.sent_num), -1).float()
        sent_sum_prb = self.sent_classification(outs_enc)
        sent_sum_prb = nn.functional.softmax(sent_sum_prb, dim=1)
        sent_sum_prb = torch.mul(sent_sum_prb, sent_num_mask)
        """_, top_k_index = torch.topk(sent_sum_prb, k=k, dim=1)
        top_k_index = torch.squeeze(top_k_index)
        top_k_mask = index_mask(self.batch_size, self.sent_num, top_k_index)
        top_k_mask = torch.unsqueeze(top_k_mask, dim=-1).cuda()
        outs_enc_filter = outs_enc.mul(top_k_mask)"""

        return sent_sum_prb

    def forward(self, k, inp_src, inp_sim, inp_trg, sim_len, sent_len, sent_num, trg_lengths):
        """
        enc1------------------>dec1
        |                        |
        |                        |
        summary------>enc2---->dec2

        (extrative-based summarization)

        inp_src: input source (batch x sent_num x sent_len)
        inp_sim: k similar sentences to nth sentence (batch x k x sim_len)
        inp_trg: input nsent (batch x nsent_len)
        sim_len: length of each sentence of similar sentences
        sent_len: length of each sentence in a dialogue
        sent_num: sentence number in a dialogue
        trg_lenghts: nth sentence length
        """

        # --------------------------------------------
        # ENCODER (encode each sentence)
        # --------------------------------------------
        # encode dialogue
        imp_src_org = inp_src.view(self.batch_size, self.sent_num, self.sent_len)
        inp_src = imp_src_org.view(imp_src_org.size(0) * imp_src_org.size(1), imp_src_org.size(2))
        inp_length = imp_src_org.size(2)
        sent_len = sent_len.view(self.batch_size * self.sent_num)
        enc1_results = self.inp_encoder(inp_src, None, sent_len, word_dropout=self.enc_token_dropout)
        outs_enc, hn_enc = enc1_results[-2:]

        sent_len_mask = torch.unsqueeze(sequence_mask(sent_len, max_len=self.sent_len), -1).float()
        outs_enc = torch.mul(outs_enc, sent_len_mask)
        outs_enc = outs_enc.view(self.batch_size, self.sent_num, self.sent_len, -1)
        outs_enc = torch.sum(outs_enc, dim=2)

        sent_num_mask = torch.unsqueeze(sequence_mask(sent_num, max_len=self.sent_num), -1).float()
        sent_sum_prb = self.sent_classification(outs_enc)
        sent_sum_prb = nn.functional.softmax(sent_sum_prb, dim=1)
        sent_sum_prb = torch.mul(sent_sum_prb, sent_num_mask)
        _, top_k_index = torch.topk(sent_sum_prb, k=k, dim=1)
        top_k_index = torch.squeeze(top_k_index)
        top_k_mask = index_mask(self.batch_size, self.sent_num, top_k_index)
        top_k_mask = torch.unsqueeze(top_k_mask, dim=-1).cuda()
        outs_enc_filter = outs_enc.mul(top_k_mask)

        # encode k similar sentences to nth sentence
        k_num = sim_len.size(1)
        inp_sim = inp_sim.view(self.batch_size, k_num, -1)
        inp_sim = inp_sim.view(self.batch_size * k_num, -1)
        sim_len = sim_len.view(self.batch_size * k_num)
        enc2_results = self.inp_encoder(inp_sim, None, sim_len, word_dropout=self.enc_token_dropout)
        outs_enc_sim, hn_enc_sim = enc2_results[-2:]
        outs_enc_sim = torch.sum(outs_enc_sim, dim=1)
        outs_enc_sim = outs_enc_sim.view(self.batch_size, k_num, -1)

        ## initiate decoder
        hn_enc_rst = []
        for index, hn_emc_tmp in enumerate(hn_enc):
            hn_emc_tmp = hn_emc_tmp.chunk(self.batch_size, dim=1)
            rst = []
            for _, sample in enumerate(hn_emc_tmp):
                sample = torch.sum(sample, dim=1)
                rst.append(sample)
            hn_emc_tmp = torch.stack(rst, dim=1)
            hn_enc_rst.append(hn_emc_tmp)
        hn_enc = tuple(hn_enc_rst)

        sent_len= sent_len.view(self.batch_size, self.sent_num)
        _dec_init = self._bridge(self.src_bridge, hn_enc, sent_num, trg_lengths)

        # -------------------------------------------------------------
        # DECODER-1 (generate nth sentence based on original dialogue)
        # -------------------------------------------------------------
        dec1_results = self.dia_nsent(inp_trg, outs_enc, _dec_init,
                                       enc_lengths=sent_num,
                                       sampling_prob=1.,
                                       desired_lengths=trg_lengths)
 
        # --------------------------------------------------
        # DECODER-2 (generate nth sentence based on summary)
        # --------------------------------------------------
        dec2_results = self.sum_nsent(inp_trg, outs_enc_filter, _dec_init,
                                         enc_lengths=sent_num,
                                         sampling_prob=1.,
                                         desired_lengths=trg_lengths)
                                       
        # --------------------------------------------------
        # Predict similar sentences
        # --------------------------------------------------
        outs_enc_pre = torch.unsqueeze(torch.sum(outs_enc, dim=1), dim=1)
        outs_enc_filter_pre = torch.unsqueeze(torch.sum(outs_enc_filter, dim=1), dim=1)

        outs_enc_pre = outs_enc_pre.expand(outs_enc_sim.size(0), outs_enc_sim.size(1), outs_enc_sim.size(2))
        outs_enc_filter_pre = outs_enc_filter_pre.expand(outs_enc_sim.size(0), outs_enc_sim.size(1), outs_enc_sim.size(2))
        outs_enc_pre = torch.cat((outs_enc_pre, outs_enc_sim), dim=-1)
        outs_enc_filter_pre = torch.cat((outs_enc_filter_pre, outs_enc_sim), dim=-1)

        outs_enc_pre = self.sent_similar(outs_enc_pre)
        outs_enc_filter_pre = self.sent_similar(outs_enc_filter_pre)

        dialog_pre = torch.squeeze(outs_enc_pre, dim=-1)
        summary_pre = torch.squeeze(outs_enc_filter_pre, dim=-1)

        return sent_sum_prb, outs_enc, outs_enc_filter, dec1_results, dec2_results, sent_len, dialog_pre, summary_pre
