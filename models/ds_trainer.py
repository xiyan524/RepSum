import torch
import numpy as np
import torch.nn as nn
from torch.nn import functional as F

#from models.seq3_losses import _kl_div, kl_length, pairwise_loss
from models.ds_losses import _kl_div, kl_length, pairwise_loss
from models.ds_utils import sample_lengths
#from models.seq3_utils import sample_lengths
from modules.helpers import sequence_mask, avg_vectors, module_grad_wrt_loss, kl_categorical
from modules.training.trainer import Trainer


class DsTrainer(Trainer):

    def __init__(self, *args, **kwargs):

        super().__init__(*args, **kwargs)

        self.oracle = kwargs.get("oracle", None)
        self.top = self.config["model"]["top"]
        self.hard = self.config["model"]["hard"]
        self.sampling = self.anneal_init(self.config["model"]["sampling"])
        self.tau = self.anneal_init(self.config["model"]["tau"])
        self.len_min_rt = self.anneal_init(self.config["model"]["min_ratio"])
        self.len_max_rt = self.anneal_init(self.config["model"]["max_ratio"])
        self.len_min = self.anneal_init(self.config["model"]["min_length"])
        self.len_max = self.anneal_init(self.config["model"]["max_length"])

    def _debug_grads(self):
        return list(sorted([(n, p.grad) for n, p in
                            self.model.named_parameters() if p.requires_grad]))

    def _debug_grad_norms(self, reconstruct_loss, prior_loss, topic_loss, kl_loss):
        c_grad_norm = []
        c_grad_norm.append(
            module_grad_wrt_loss(self.optimizers, self.model.compressor,
                                 reconstruct_loss,
                                 "rnn"))

        if self.config["model"]["topic_loss"]:
            c_grad_norm.append(
                module_grad_wrt_loss(self.optimizers, self.model.compressor,
                                     topic_loss,
                                     "rnn"))

        if self.config["model"]["prior_loss"] and self.oracle is not None:
            c_grad_norm.append(
                module_grad_wrt_loss(self.optimizers, self.model.compressor,
                                     prior_loss,
                                     "rnn"))

        if self.config["model"]["doc_sum_kl_loss"] and self.oracle is not None:
            c_grad_norm.append(
                module_grad_wrt_loss(self.optimizers, self.model.compressor,
                                     kl_loss,
                                     "rnn"))

        return c_grad_norm

    def _topic_loss(self, inp, dec1, src_lengths, trg_lengths):
        """
        Compute the pairwise distance of various outputs of the seq^3 architecture.
        Args:
            enc1: the outputs of the first encoder (input sequence)
            dec1: the outputs of the first decoder (latent sequence)
            src_lengths: the lengths of the input sequence
            trg_lengths: the lengths of the targer sequence (summary)

        """

        enc_mask = sequence_mask(src_lengths).unsqueeze(-1).float()
        dec_mask = sequence_mask(trg_lengths - 1).unsqueeze(-1).float()

        enc_embs = self.model.inp_encoder.embed(inp)
        dec_embs = self.model.compressor.embed.expectation(dec1[3])

        if self.config["model"]["topic_idf"]:
            enc1_energies = self.model.idf(inp)
            # dec1_energies = expected_vecs(dec1[3], self.model.idf.weight)

            x_emb, att_x = avg_vectors(enc_embs, enc_mask, enc1_energies)
            # y_emb, att_y = avg_vectors(dec_reps, dec_mask, dec1_energies)
            y_emb, att_y = avg_vectors(dec_embs, dec_mask)

        else:
            x_emb, att_x = avg_vectors(enc_embs, enc_mask)
            y_emb, att_y = avg_vectors(dec_embs, dec_mask)

        distance = self.config["model"]["topic_distance"]
        loss = pairwise_loss(x_emb, y_emb, distance)

        return loss, (att_x, att_y)

    #def _doc_sum_loss(self, enc1, enc2, doc_lengths, sum_lengths):
    def _doc_sum_loss(self, inp, attn_dis, src_lengths, trg_lengths):
        """
        Compute the loss of semantic representation between document and summary
        Args:
            enc1: the outputs of the first encoder (input sequence)
            enc2: the outputs of the first decoder (decode summary)
        """

        """doc_mask = sequence_mask(doc_lengths).unsqueeze(-1).float()
        sum_mask = sequence_mask(sum_lengths - 1).unsqueeze(-1).float()

        doc_vec = enc1[0] * doc_mask.float()
        doc_mean = doc_vec.sum(1) / doc_mask.sum(1)

        sum_vec = enc2[0] * sum_mask.float()
        sum_mean = sum_vec.sum(1) / sum_mask.sum(1)
        
        loss = torch.cosine_similarity(doc_mean, sum_mean)
        loss = torch.mean(torch.abs(loss))"""

        enc_mask = sequence_mask(src_lengths).unsqueeze(-1).float()
        enc_embs = self.model.inp_encoder.embed(inp)

        if self.config["model"]["topic_idf"]:
            enc1_energies = self.model.idf(inp)
            x_emb, att_x = avg_vectors(enc_embs, enc_mask, enc1_energies)
        #tmp = attn_dis.contiguous().view(-1, attn_dis.size(-1))
        attn_dis = torch.sum(attn_dis, 1)
        att_x = torch.squeeze(att_x)

        #loss = torch.dist(att_x, attn_dis, 1)
        distance = self.config["model"]["docsim_distance"]
        loss = pairwise_loss(att_x, attn_dis)

        return loss

    def _prior_loss(self, outputs, latent_lengths):
        """
        Prior Loss
        Args:
            outputs:
            latent_lengths:

        Returns:

        """
        enc1, dec1, enc2, dec2, dec3 = outputs
        _vocab = self._get_vocab()

        logits_dec1, outs_dec1, hn_dec1, dists_dec1, _, _ = dec1

        # dists_dec1 contain the distributions from which
        # the samples were taken. It contains one less element than the logits
        # because the last logit is only used for computing the NLL of EOS.
        words_dec1 = dists_dec1.max(-1)[1]

        # sos + the sampled sentence
        sos_id = _vocab.tok2id[_vocab.SOS]
        sos = torch.zeros_like(words_dec1[:, :1]).fill_(sos_id)
        oracle_inp = torch.cat([sos, words_dec1], -1)

        logits_oracle, _, _ = self.oracle(oracle_inp, None,
                                          latent_lengths)

        prior_loss, prior_loss_time = _kl_div(logits_dec1,
                                              logits_oracle,
                                              latent_lengths)

        return prior_loss, prior_loss_time, logits_oracle

    def _process_batch(self, inp_x, inp_sim, out_sim, inp_y, out_y, sim_len, sent_len, sent_num, y_lengths):

        self.model.train()

        outputs = self.model(self.config["model"]["k"], inp_x, inp_sim, inp_y, sim_len, sent_len, sent_num, y_lengths)

        sent_prob, outs_enc, outs_enc_filter, dec1, dec2, sent_len, dialogue_pre, summary_pre = outputs

        batch_outputs = {"model_outputs": outputs}

        # --------------------------------------------------------------
        # 1 - Predict nth sentence
        # --------------------------------------------------------------
        _dec1_logits = dec1[0].contiguous().view(-1, dec1[0].size(-1))
        _x_labels = out_y.contiguous().view(-1)
        nsent_loss = F.cross_entropy(_dec1_logits, _x_labels, ignore_index=0, reduction='none')
 
        nsent_loss_token = nsent_loss.view(out_y.size())
        batch_outputs["n_sent"] = nsent_loss_token
        mean_rec_loss = nsent_loss.sum() / y_lengths.float().sum()
        losses = [mean_rec_loss]

        # --------------------------------------------------------------
        # 1.5 - Predict nth sentence from summary
        # --------------------------------------------------------------
        if self.config["model"]["n_sent_sum_loss"]:
            _dec2_logits = dec2[0].contiguous().view(-1, dec2[0].size(-1))
            nsent_loss_sum = F.cross_entropy(_dec2_logits, _x_labels, ignore_index=0, reduction='none')
            nsent_loss_token_sum = nsent_loss_sum.view(out_y.size())
            batch_outputs["n_sent_sum"] = nsent_loss_token_sum
            mean_rec_sum_loss = nsent_loss_sum.sum() / y_lengths.float().sum()
            losses.append(mean_rec_sum_loss)
        else:
            mean_rec_sum_loss = None

        # --------------------------------------------------------------
        # 2 - DOCUEMNT+SUMMARY DISTRIBUTION
        # --------------------------------------------------------------
        if self.config["model"]["doc_sum_kl_loss"]:
            _dec1_logits = dec1[0].contiguous().view(-1, dec1[0].size(-1))
            _dec2_logits = dec2[0].contiguous().view(-1, dec2[0].size(-1))
            #kl_loss = torch.nn.functional.kl_div(_dec2_logits, _dec2_logits, size_average=None, reduce=True, reduction='mean')
            kl_loss = kl_categorical(_dec1_logits, _dec2_logits)
            losses.append(kl_loss)
        else:
            kl_loss = None

        # --------------------------------------------------------------
        # 3 - LENGTH 
        # --------------------------------------------------------------
        if self.config["model"]["length_loss"]:
            _, topk_indices = torch.topk(sent_prob, k=self.config["model"]["k"], dim=1)
            topk_indices = torch.squeeze(topk_indices, -1)
            sum_length = torch.gather(sent_len, dim=1, index=topk_indices)
            sum_length = torch.sum(sum_length, dim=1)
            tmp = torch.sub(self.config["data"]["ext_sum_len"], sum_length)
            length_loss = torch.mean(tmp.float())
            losses.append(length_loss)
        else:
            length_loss = None

        # --------------------------------------------------------------
        # 4 - DOCUEMNT SUMMARY SIMILARITY
        # --------------------------------------------------------------
        if self.config["model"]["doc_sum_sim_loss"]:
            dialog_rep = torch.squeeze(torch.sum(outs_enc, 1))
            summary_rep = torch.squeeze(torch.sum(outs_enc_filter, 1))
            sim_loss = pairwise_loss(dialog_rep, summary_rep)
            losses.append(sim_loss)
        else:
            sim_loss = None

        # --------------------------------------------------------------
        # 4 - N SENTENCE CLASSIFICATION
        # --------------------------------------------------------------
        if self.config["model"]["nsent_classification"]:
            criterion = torch.nn.BCEWithLogitsLoss()
            dia_pre_loss = criterion(dialogue_pre, out_sim.float())
            sum_pre_loss = criterion(summary_pre, out_sim.float())
            pre_kl_loss = kl_categorical(dialogue_pre, summary_pre)
            losses.append(dia_pre_loss)
            losses.append(sum_pre_loss)
            losses.append(pre_kl_loss)
        else:
            dia_pre_loss = None
            sum_pre_loss = None
            pre_kl_loss = None
            
        prior_loss = None
        topic_loss = None
        kl_loss = None
        # --------------------------------------------------------------
        # Plot Norms of loss gradient wrt to the compressor
        # --------------------------------------------------------------
        if self.config["plot_norms"] and self.step % self.config["log_interval"] == 0:
            batch_outputs["grad_norm"] = self._debug_grad_norms(
                mean_rec_loss,
                prior_loss,
                topic_loss,
                kl_loss)

        return losses, batch_outputs

    def eval_epoch(self, batch_size):
        """
        Evaluate the network for one epoch and return the average loss.

        Returns:
            loss (float, list(float)): list of mean losses

        """
        self.model.eval()

        results = []
        k_indices = []
        oov_maps = []

        self.len_min_rt = self.anneal_init(
            self.config["model"]["test_min_ratio"])
        self.len_max_rt = self.anneal_init(
            self.config["model"]["test_max_ratio"])
        self.len_min = self.anneal_init(
            self.config["model"]["test_min_length"])
        self.len_max = self.anneal_init(
            self.config["model"]["test_max_length"])

        iterator = self.valid_loader
        with torch.no_grad():
            for i_batch, batch in enumerate(iterator, 1):
                batch_oov_map = batch[-1]
                batch = batch[:-1]

                batch = list(map(lambda x: x.to(self.device), batch))
                (inp_src, inp_sim, out_sim, out_src, out_trg, sim_len, sent_len, sent_num, trg_lengths) = batch

                if inp_src.size()[0] != batch_size:
                    continue

                sent_prob = self.model.summary(inp_src, sent_len, sent_num)
                sent_prob = torch.squeeze(sent_prob)
                _, topk_indices = torch.topk(sent_prob, k=self.config["model"]["k"], dim=1)
                inp_src = inp_src.view(inp_src.size(0), self.config["model"]["sent_num"], self.config["model"]["sent_len"])

                inp_src = inp_src.chunk(batch_size, dim=0)
                topk_indices = topk_indices.chunk(batch_size, dim=0)
                result_sents = []
                for inp, indice in zip(inp_src, topk_indices):
                    inp = torch.squeeze(inp)
                    indice = torch.squeeze(indice)
                    sum_sent = torch.index_select(inp, 0, indice)
                    result_sents.append(sum_sent)

                oov_maps.append(batch_oov_map)
                results.append(result_sents)
                k_indices.append(topk_indices)

        return results, oov_maps, k_indices

    def _get_vocab(self):
        if isinstance(self.train_loader, (list, tuple)):
            dataset = self.train_loader[0].dataset
        else:
            dataset = self.train_loader.dataset

        if dataset.subword:
            _vocab = dataset.subword_path
        else:
            _vocab = dataset.vocab

        return _vocab

    def get_state(self):

        state = {
            "config": self.config,
            "epoch": self.epoch,
            "step": self.step,
            "model": self.model.state_dict(),
            "model_class": self.model.__class__.__name__,
            "optimizers": [x.state_dict() for x in self.optimizers],
            "vocab": self._get_vocab(),
        }

        return state
