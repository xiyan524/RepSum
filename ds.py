import itertools
import math
import os
import warnings

import numpy
import codecs
import torch
from tabulate import tabulate
from torch import nn
from torch.distributions import Categorical
from torch.utils.data import DataLoader
from generate.utils import devectorize, devectorize_generate
from models.ds_trainer import DsTrainer
from models.ds_utils import compute_dataset_idf
from modules.data.collates import Seq2SeqCollate, Seq2SeqOOVCollate
from modules.data.datasets_ds import DsDataset
from modules.data.samplers import BucketBatchSampler
from modules.models import Seq2Seq2Seq
from modules.modules import SeqReader, transfer_weigths
from mylogger.attention import samples2html
from mylogger.experiment import Experiment
from sys_config import EXP_DIR, EMBS_PATH, MODEL_CNF_DIR
from utils.eval import rouge_file_list, pprint_rouge_scores, rouge_files, rouge_files_simple
from utils.generic import number_h
from utils.opts import seq2seq2seq_options
from utils.training import load_checkpoint
from utils.transfer import freeze_module
from tensorboardX import SummaryWriter

####################################################################
# Settings
####################################################################
opts, config = seq2seq2seq_options()
folder = os.path.exists(config["data"]["result_path"])
if not folder:  
    os.makedirs(config["data"]["result_path"]) 
vocab = None
writer = SummaryWriter(config["tensorboard_path"])

if config["main_model"] is not None:
    main_model_checkpoint = load_checkpoint(config["main_model"])


####################################################################
# Weight Transfer (pre-train language model)
####################################################################
if config["model"]["prior_loss"] and config["prior"] is not None:
    print("Loading Oracle LM ...")
    oracle_cp = load_checkpoint(config["prior"])
    vocab = oracle_cp["vocab"]

    oracle = SeqReader(len(vocab), **oracle_cp["config"]["model"])
    oracle.load_state_dict(oracle_cp["model"])
    oracle.to(opts.device)
    freeze_module(oracle)
else:
    oracle = None


####################################################################
# Data Loading and Preprocessing
####################################################################
print("Building training dataset...")
train_data = DsDataset(config["data"]["train_path"],
                       batch_size=config["batch_size"],
                       sent_num=config["data"]["sent_num"],
                       sent_len=config["data"]["sent_len"],
                       mode="train",
                       sent_sim_len=config["data"]["sent_sim_len"],
                       k=config["model"]["k"],
                       vocab=vocab,
                       vocab_size=config["vocab"]["size"],
                       seq_len=config["data"]["seq_len"],
                       dec_seq_len= config["data"]["dec_seq_len"],
                       oovs=config["data"]["oovs"])


print("Building validation dataset...")
val_data = DsDataset(config["data"]["val_path"],
                       batch_size=config["batch_size"],
                       sent_num=config["data"]["sent_num"],
                       sent_len=config["data"]["sent_len"],
                       mode="test",
                       k=config["model"]["k"],
                       sent_sim_len=config["data"]["sent_sim_len"],
                       summary_path=config["data"]["ref_path"],
                       nsent_path=config["data"]["ref_nsent_path"],
                       org_dia_path=config["data"]["org_dia_path"],
                       vocab=vocab,
                       vocab_size=config["vocab"]["size"],
                       seq_len=config["data"]["seq_len"],
                       dec_seq_len= config["data"]["dec_seq_len"],
                       return_oov=True,
                       oovs=config["data"]["oovs"])

val_data.vocab = train_data.vocab
vocab = train_data.vocab

# define a dataloader, which handles the way a dataset will be loaded,
# like batching, shuffling and so on ...
train_lengths = [len(x) for x in train_data.data]

train_sampler = BucketBatchSampler(train_lengths, config["batch_size"])
train_loader = DataLoader(train_data, batch_sampler=train_sampler,
                          num_workers=config["num_workers"],
                          collate_fn=Seq2SeqCollate(config["data"]["sent_sim_len"], config["data"]["sent_num"], config["data"]["sent_len"]))
val_loader = DataLoader(val_data, batch_size=config["batch_size"],
                        num_workers=config["num_workers"], shuffle=False,
                        collate_fn=Seq2SeqOOVCollate(config["data"]["sent_sim_len"], config["data"]["sent_num"], config["data"]["sent_len"]))

####################################################################
# Model Definition
# - additional layer initializations
# - weight / layer tying
####################################################################

# Define the model
n_tokens = len(train_data.vocab)
model = Seq2Seq2Seq(n_tokens, **config["model"])
criterion = nn.CrossEntropyLoss(ignore_index=0)

def word_embedding(model):
    # Load Pretrained Word Embeddings
    if "embeddings" in config["vocab"] and config["vocab"]["embeddings"]:
        emb_file = os.path.join(EMBS_PATH, config["vocab"]["embeddings"])
        dims = config["vocab"]["embeddings_dim"]

        embs, emb_mask, missing = train_data.vocab.read_embeddings(emb_file, dims)
        model.initialize_embeddings(embs, config["model"]["embed_trainable"])

        # initialize the output layers with the pretrained embeddings,
        # regardless of whether they will be tied
        try:
            model.compressor.Wo.weight.data.copy_(torch.from_numpy(embs))
            model.decompressor.Wo.weight.data.copy_(torch.from_numpy(embs))
        except:
            print("Can't init outputs from embeddings. Dim mismatch!")

        if config["model"]["embed_masked"] and config["model"]["embed_trainable"]:
            model.set_embedding_gradient_mask(emb_mask)


def topic_pre(model):
    if config["model"]["topic_loss"] and config["model"]["topic_idf"]:
        print("Computing IDF values...")
    idf = compute_dataset_idf(train_data, train_data.vocab.tok2id)
    # idf[vocab.tok2id[vocab.SOS]] = 1  # neutralize padding token
    # idf[vocab.tok2id[vocab.EOS]] = 1  # neutralize padding token
    idf[vocab.tok2id[vocab.PAD]] = 1  # neutralize padding token
    model.initialize_embeddings_idf(idf)


def tie_models(model):
    """Tie encoder/decoder of models"""

    # tie the embedding layers
    if config["model"]["tie_embedding"]:
        model.cmp_encoder.embed = model.inp_encoder.embed
        model.compressor.embed = model.inp_encoder.embed
        model.decompressor.embed = model.inp_encoder.embed
        model.original_task.embed = model.inp_encoder.embed

    # tie the output layers of the decoders
    """if config["model"]["tie_decoder_outputs"]:
        model.compressor.Wo = model.decompressor.Wo"""

    # tie the embedding to the output layers
    if config["model"]["tie_embedding_outputs"]:
        emb_size = model.compressor.embed.embedding.weight.size(1)
        rnn_size = model.compressor.Wo.weight.size(1)

        if emb_size != rnn_size:
            warnings.warn("Can't tie outputs, since emb_size != rnn_size.")
        else:
            model.compressor.Wo.weight = model.inp_encoder.embed.embedding.weight
            model.decompressor.Wo.weight = model.inp_encoder.embed.embedding.weight
            model.original_task.Wo.weight = model.inp_encoder.embed.embedding.weight

    if config["model"]["tie_decoders"]:
        #model.compressor = model.decompressor
        #model.decompressor = model.original_task
        transfer_weigths(model.decompressor, model.original_task)
        transfer_weigths(model.compressor, model.original_task)
        
    if config["model"]["tie_encoders"]:
        #model.cmp_encoder = model.inp_encoder
        transfer_weigths(model.cmp_encoder, model.inp_encoder)

    # then we need only one bridge
    if config["model"]["tie_encoders"] and config["model"]["tie_decoders"]:
        model.src_bridge = model.trg_bridge


word_embedding(model)
#topic_pre(model)
tie_models(model)

####################################################################
# Experiment Logging and Visualization
####################################################################
parameters = filter(lambda p: p.requires_grad, model.parameters())
tmp = model.parameters()
optimizer = torch.optim.Adam(parameters,
                             lr=config["lr"],
                             weight_decay=config["weight_decay"])

model.to(opts.device)
print(model)

total_params = sum(p.numel() for p in model.parameters())
total_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

print("Total Params:", number_h(total_params))
print("Total Trainable Params:", number_h(total_trainable_params))
trainable_params = sorted([[n] for n, p in model.named_parameters() if p.requires_grad])


def exp_log_visual(model):
    """Experiment Logging and Visualization"""
    if config["prior"] is not None:
        opts.name += "_" + config["prior"]

    exp = Experiment(opts.name, config, src_dirs=opts.source, output_dir=EXP_DIR)

    step_tags = []
    step_tags.append("NSENT")

    if config["model"]["n_sent_sum_loss"]:
        step_tags.append("NSENTSUM")
    if config["model"]["prior_loss"] and config["prior"] is not None:
        step_tags.append("PRIOR")
    if config["model"]["topic_loss"]:
        step_tags.append("TOPIC")
    if config["model"]["length_loss"]:
        step_tags.append("LENGTH")
    if config["model"]["doc_sum_kl_loss"]:
        step_tags.append("DOCSUMKL")
    if config["model"]["doc_sum_sim_loss"]:
        step_tags.append("DOCSUMSIM")
    if config["model"]["sum_loss"]:
        step_tags.append("SUMMARY")
    if config["model"]["nsent_classification"]:
        step_tags.append("CLS")
    if config["model"]["nsent_classification_sum"]:
        step_tags.append("CLSSUM")
    if config["model"]["nsent_classification_kl"]:
        step_tags.append("CLSKL")
    
    exp.add_metric("loss", "line", tags=step_tags)
    exp.add_metric("ppl", "line", title="perplexity", tags=step_tags)
    exp.add_metric("rouge", "line", title="ROUGE (F1)", tags=["R-1", "R-2", "R-L"])
    exp.add_value("grads", "text", title="gradients")

    exp.add_metric("c_norm", "line", title="Compressor Grad Norms",
                tags=step_tags[:len(set(step_tags) & {"NSENTSUM", "PRIOR", "TOPIC", "DOCSUMKL", "DOCSUMSIM", "SUMMARY", 
                "CLS", "CLSSUM", "CLSKL"}) + 1])
    exp.add_value("progress", "text", title="training progress")
    exp.add_value("epoch", "text", title="epoch summary")
    exp.add_value("samples", "text", title="Samples")
    exp.get_value("samples").pre = False
    exp.add_value("weights", "text")
    exp.add_value("rouge-stats", "text")
    exp.add_value("states", "scatter")
    exp.add_metric("lr", "line", "Learning Rate")
    exp.add_value("rouge-stats", "text")

    return exp, step_tags

exp, step_tags = exp_log_visual(model)

####################################################################
#
# Training Pipeline
# - batch/epoch callbacks for logging, checkpoints, visualization...
# - initialize trainer
# - initialize training loop
#
####################################################################
def stats_callback(batch, losses, loss_list, batch_outputs, epoch):
    if trainer.step % config["log_interval"] == 0:

        # log gradient norms
        grads = sorted(trainer.grads(), key=lambda tup: tup[1], reverse=True)
        grads_table = tabulate(grads, numalign="right", floatfmt=".5f", headers=['Parameter', 'Grad(Norm)'])
        exp.update_value("grads", grads_table)

        _losses = losses[-config["log_interval"]:]
        mono_losses = numpy.array([x[:len(step_tags)] for x in _losses]).mean(0)
        for loss, tag in zip(mono_losses, step_tags):
            exp.update_metric("loss", loss, tag)
            exp.update_metric("ppl", math.exp(loss), tag)

        ################################################
        losses_log = exp.log_metrics(["loss", "ppl"], epoch)
        exp.update_value("progress", trainer.progress_log + "\n" + losses_log)

        # clean lines and move cursor back up N lines
        print("\n\033[K" + losses_log)
        #print("\033[F" * (len(losses_log.split("\n")) + 2))


def samples_to_text(tensor):
    return devectorize(tensor.tolist(), train_data.vocab.id2tok,
                       train_data.vocab.tok2id[vocab.EOS],
                       strip_eos=False, pp=False)


def outs_callback(batch, losses, loss_list, batch_outputs, epoch):
    if trainer.step % config["log_interval"] == 0:
        prob, enc, enc_filter, dec1, dec2, sent_num, dialog_pre, summary_pre = batch_outputs['model_outputs']

        if config["plot_norms"]:
            norms = batch_outputs['grad_norm']
            exp.update_metric("c_norm", norms[0], "NSENT")

            if "NSENTSUM" in step_tags:
                exp.update_metric("c_norm", norms[loss_ids["nsent_sum"]], "NSENTSUM")

            if "TOPIC" in step_tags:
                exp.update_metric("c_norm", norms[loss_ids["topic"]], "TOPIC")

            if "PRIOR" in step_tags:
                exp.update_metric("c_norm", norms[loss_ids["prior"]], "PRIOR")
            
            if "DOC-SUM" in step_tags:
                exp.update_metric("c_norm", norms[loss_ids["doc_sum_kl"]], "DOCSUM")

            if "SUMMARY" in step_tags:
                exp.update_metric("c_norm", norms[loss_ids["sum"]], "SUMMARY")
            
            if "CLS" in step_tags:
                exp.update_metric("c_norm", norms[loss_ids["cls"]], "CLS")

            if "CLSSUM" in step_tags:
                exp.update_metric("c_norm", norms[loss_ids["clssum"]], "CLSSUM")
  
            if "CLSKL" in step_tags:
                exp.update_metric("c_norm", norms[loss_ids["clskl"]], "CLSKL")
            
        if len(batch) == 2:
            inp = batch[0][0]
        else:
            inp = batch[0]
        src = samples_to_text(inp)
        hyp = samples_to_text(dec1[3].max(dim=2)[1])
        nsent = samples_to_text(dec2[0].max(dim=2)[1])

        # prior outputs
        if "prior" in batch_outputs:
            prior_loss = batch_outputs['prior'][0].squeeze().tolist()
            prior_logits = batch_outputs['prior'][1]

            prior_argmax = prior_logits.max(dim=2)[1]
            prior_entropy = Categorical(logits=prior_logits).entropy().tolist()

            prior = samples_to_text(prior_argmax)

        if "attention" in batch_outputs:
            att_scores = batch_outputs['attention'][0].squeeze().tolist()
        else:
            att_scores = None

        if config["model"]["learn_tau"]:
            temps = dec1[5].cpu().data.numpy().round(2)
        else:
            temps = None

        nsent_losses = batch_outputs['n_sent'].tolist()

        samples = []
        for i in range(len(src)):
            sample = []

            if att_scores is not None:
                _src = 'SRC', (src[i], att_scores[i]), "255, 0, 0"
            else:
                _src = 'SRC', src[i], "0, 0, 0"
            sample.append(_src)

            if "prior" in batch_outputs:
                _hyp = 'HYP', (hyp[i], prior_loss[i]), "0, 0, 255"
                _pri = 'LM ', (prior[i], prior_entropy[i]), "0, 255, 0"
                sample.append(_hyp)
                sample.append(_pri)
            else:
                _hyp = 'HYP', hyp[i], "0, 0, 255"
                sample.append(_hyp)

            if temps is not None:
                _tmp = 'TMP', (list(map(str, temps[i])), temps[i]), "255, 0, 0"
                sample.append(_tmp)

            _nsent = 'NSENT', (nsent[i], nsent_losses[i]), "255, 0, 0"
            sample.append(_nsent)

            samples.append(sample)

        html_samples = samples2html(samples)
        exp.update_value("samples", html_samples)
        with open(os.path.join(EXP_DIR, f"{opts.name}.samples.html"), 'w') as f:
            f.write(html_samples)

best_test_score = -1.
def eval_callback(batch, losses, loss_list, batch_outputs, epoch):
    global best_test_score
    if trainer.step % config["checkpoint_interval"] == 0:
        tags = [trainer.epoch, trainer.step]
        trainer.checkpoint(name=opts.name, tags=tags)
        exp.save()

    original_dialogue = val_data.inputs
    if trainer.step % config["eval_interval"] == 0:
        results_sent, oov_maps, k_indices = trainer.eval_epoch(config["batch_size"])
        results_sent = list(itertools.chain.from_iterable(results_sent))
        oov_maps = list(itertools.chain.from_iterable(oov_maps))

        # generate prund summary
        v = train_data.vocab
        tokens = devectorize_generate(results_sent, v.id2tok, v.tok2id[v.EOS], True, oov_maps)
        hyps = [" ".join(x) for x in tokens]

        # generate original summary
        hyps_org = []
        k_indices = [x for j in k_indices for x in j]
        for index in range(len(k_indices)):
            try:
                k_sent_index = k_indices[index].tolist()
                tmp_file = ""
                for j in k_sent_index[0]:
                    tmp = original_dialogue[index]
                    tmp_file += original_dialogue[index][j] + " "
                hyps_org.append(tmp_file)
            except Exception:
                hyps_org.append("被告")

        # evaluate summary
        #scores = rouge_files(config["data"]["result_path"], config["data"]["ref_path"], hyps)
        scores = rouge_files(config["data"]["result_path"], config["data"]["ref_path"], hyps_org)
        rouge_table = pprint_rouge_scores(scores)
        exp.update_value("rouge-stats", rouge_table)
        exp.update_metric("rouge", scores['rouge-1']['f'], "R-1")
        exp.update_metric("rouge", scores['rouge-2']['f'], "R-2")
        exp.update_metric("rouge", scores['rouge-l']['f'], "R-L")
  
        epoch_times = trainer.step / config["eval_interval"]
        writer.add_scalar('Test/rouge-1', scores['rouge-1']['f'], epoch_times)
        writer.add_scalar('Test/rouge-2', scores['rouge-2']['f'], epoch_times)
        writer.add_scalar('Test/rouge-l', scores['rouge-l']['f'], epoch_times)
        writer.flush()

        if scores['rouge-1']['f'] > best_test_score:
            best_test_score = scores['rouge-1']['f']
            # save the best decode results (base on rouge1-f)
            tmp_decode_file = codecs.open(config["data"]["dec_summ_path"], 'w')
            for hyp in hyps_org:
                tmp_decode_file.write(hyp.replace("\n", "")+"\n")

        save_best()


####################################################################
# Loss Weight: order matters!
####################################################################
def loss_id():
    loss_ids = {}

    loss_weights = [config["model"]["loss_weight_nsent"]]
    loss_ids["nsent"] = len(loss_weights) - 1
    if config["model"]["n_sent_sum_loss"]:
        loss_weights.append(config["model"]["loss_weight_nsent_sum"])
        loss_ids["nsent_sum"] = len(loss_weights) - 1
    if config["model"]["prior_loss"] and config["prior"] is not None:
        loss_weights.append(config["model"]["loss_weight_prior"])
        loss_ids["prior"] = len(loss_weights) - 1
    if config["model"]["topic_loss"]:
        loss_weights.append(config["model"]["loss_weight_topic"])
        loss_ids["topic"] = len(loss_weights) - 1
    if config["model"]["length_loss"]:
        loss_weights.append(config["model"]["loss_weight_length"])
        loss_ids["length"] = len(loss_weights) - 1
    if config["model"]["doc_sum_kl_loss"]:
        loss_weights.append(config["model"]["loss_weight_doc_sum"])
        loss_ids["doc_sum_kl"] = len(loss_weights) - 1
    if config["model"]["doc_sum_sim_loss"]:
        loss_weights.append(config["model"]["loss_weight_doc_sum_sim"])
        loss_ids["doc_sum_sim"] = len(loss_weights) - 1
    if config["model"]["sum_loss"]:
        loss_weights.append(config["model"]["loss_weight_sum"])
        loss_ids["sum"] = len(loss_weights) - 1
    if config["model"]["nsent_classification"]:
        loss_weights.append(config["model"]["loss_weight_classification"])
        loss_ids["cls"] = len(loss_weights) - 1
    if config["model"]["nsent_classification_sum"]:
        loss_weights.append(config["model"]["loss_weight_classification_sum"])
        loss_ids["cls_sum"] = len(loss_weights) - 1
    if config["model"]["nsent_classification_kl"]:
        loss_weights.append(config["model"]["loss_weight_classification_kl"])
        loss_ids["cls_sum"] = len(loss_weights) - 1
    return loss_id, loss_weights

loss_id, loss_weights = loss_id()
if config["main_model"] is not None:
    #optimizer.load_state_dict(main_model_checkpoint["optimizers"])
    model.load_state_dict(main_model_checkpoint["model"])
trainer = DsTrainer(model, train_loader, val_loader,
                      criterion, optimizer, config, opts.device,
                      batch_end_callbacks=[stats_callback,
                                           outs_callback,
                                           eval_callback],
                      loss_weights=loss_weights, oracle=oracle)

####################################################################
# Training Loop
####################################################################

assert not train_data.vocab.is_corrupt()
assert not val_data.vocab.is_corrupt()

best_score = None
def save_best():
    global best_score
    _score = exp.get_metric("rouge").values["R-2"][-1]
    if not best_score or _score > best_score:
        best_score = _score
        trainer.checkpoint()
    exp.save()

for epoch in range(config["epochs"]):
    batch_num = train_data.__len__()/config["batch_size"]
    train_loss = trainer.train_epoch(config["pre_train_epochs"], batch_num, writer)

    # Save the model if the validation loss is the best we've seen so far.
    save_best()
