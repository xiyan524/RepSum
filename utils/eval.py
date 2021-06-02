import pandas

import rouge
import codecs
import os
import files2rouge
from tabulate import tabulate

def rouge_lists(refs, hyps):
    evaluator = rouge.Rouge(metrics=['rouge-n', 'rouge-l'],
                            max_n=2,
                            limit_length=True,
                            length_limit=100,
                            length_limit_type='words',
                            apply_avg=True,
                            apply_best=False,
                            alpha=0.5,  # Default F1_score
                            weight_factor=1.2,
                            stemming=True)
    scores = evaluator.get_scores(hyps, refs)

    return scores


def tokens_to_ids(token_list1, token_list2):
    ids = {}
    out1 = []
    out2 = []
    for token in token_list1:
        out1.append(ids.setdefault(token, len(ids)))
    for token in token_list2:
        out2.append(ids.setdefault(token, len(ids)))
    return out1, out2


def write_id(id_lst, file):
    for id in id_lst:
        file.write(str(id)+" ")
    file.write("\n")


def filter_file(file_lst):
    if "" == file_lst[-1]:
        del(file_lst[-1])
    return file_lst


# trans word to id to escape from chinese character
def trans_id(path, hyps, ref_path):
    #dec_files = codecs.open(dec_path, encoding="utf-8").read().split("\n")
    dec_files = hyps
    ref_files = codecs.open(ref_path, encoding="utf-8").read().split("\n")
    dec_files = filter_file(dec_files)
    ref_files = filter_file(ref_files)

    dec_files_id = codecs.open(os.path.join(path, "decode_id_tmp.txt"), 'w')
    ref_files_id = codecs.open(os.path.join(path, "reference_id_tmp.txt"), 'w')

    results_path = os.path.join(path, "results.txt")

    sample_num = len(dec_files)
    for index in range(sample_num):
        dec_file = dec_files[index].split(" ")
        ref_file = ref_files[index].split(" ")
        dec_id, ref_id = tokens_to_ids(dec_file, ref_file)
        write_id(dec_id, dec_files_id)
        write_id(ref_id, ref_files_id)

    scores_str = files2rouge.run(os.path.join(path, "decode_id_tmp.txt"), os.path.join(path, "reference_id_tmp.txt"))
    return scores_str


def rouge_files(path, refs_file, hyps):
    #refs = open(refs_file).readlines()
    #hyps = open(hyps_file).readlines()
    #scores = rouge_lists(refs, hyps)
    scores_str = trans_id(path, hyps, refs_file)
    result_file = codecs.open(os.path.join(path, "result.txt"), 'a')
    result_file.write(scores_str)

    r1_r = scores_str[scores_str.find("ROUGE-1 Average_R:")+19:scores_str.find("ROUGE-1 Average_R:")+26]
    r2_r = scores_str[scores_str.find("ROUGE-2 Average_R:")+19:scores_str.find("ROUGE-2 Average_R:")+26]
    rl_r = scores_str[scores_str.find("ROUGE-L Average_R:")+19:scores_str.find("ROUGE-L Average_R:")+26]

    r1_p = scores_str[scores_str.find("ROUGE-1 Average_P:")+19:scores_str.find("ROUGE-1 Average_P:")+26]
    r2_p = scores_str[scores_str.find("ROUGE-2 Average_P:")+19:scores_str.find("ROUGE-2 Average_P:")+26]
    rl_p = scores_str[scores_str.find("ROUGE-L Average_P:")+19:scores_str.find("ROUGE-L Average_P:")+26]
 
    r1_f = scores_str[scores_str.find("ROUGE-1 Average_F:")+19:scores_str.find("ROUGE-1 Average_F:")+26]
    r2_f = scores_str[scores_str.find("ROUGE-2 Average_F:")+19:scores_str.find("ROUGE-2 Average_F:")+26]
    rl_f = scores_str[scores_str.find("ROUGE-L Average_F:")+19:scores_str.find("ROUGE-L Average_F:")+26]

    scores = {}
    scores['rouge-1'] = {}
    scores['rouge-2'] = {}
    scores['rouge-l'] = {}

    scores['rouge-1']['r'] = float(r1_r)
    scores['rouge-1']['p'] = float(r1_p)
    scores['rouge-1']['f'] = float(r1_f)

    scores['rouge-2']['r'] = float(r2_r)
    scores['rouge-2']['p'] = float(r2_p)
    scores['rouge-2']['f'] = float(r2_f)

    scores['rouge-l']['r'] = float(rl_r)
    scores['rouge-l']['p'] = float(rl_p)
    scores['rouge-l']['f'] = float(rl_f)
    return scores


def rouge_files_simple(path, refs_file, hyps):
    scores_str = trans_id(path, hyps, refs_file)
    result_file = codecs.open(os.path.join(path, "result_nsent.txt"), 'a')
    result_file.write(scores_str)

    r1_r = scores_str[scores_str.find("ROUGE-1 Average_R:")+19:scores_str.find("ROUGE-1 Average_R:")+26]
    r2_r = scores_str[scores_str.find("ROUGE-2 Average_R:")+19:scores_str.find("ROUGE-2 Average_R:")+26]
    rl_r = scores_str[scores_str.find("ROUGE-L Average_R:")+19:scores_str.find("ROUGE-L Average_R:")+26]

    r1_p = scores_str[scores_str.find("ROUGE-1 Average_P:")+19:scores_str.find("ROUGE-1 Average_P:")+26]
    r2_p = scores_str[scores_str.find("ROUGE-2 Average_P:")+19:scores_str.find("ROUGE-2 Average_P:")+26]
    rl_p = scores_str[scores_str.find("ROUGE-L Average_P:")+19:scores_str.find("ROUGE-L Average_P:")+26]
 
    r1_f = scores_str[scores_str.find("ROUGE-1 Average_F:")+19:scores_str.find("ROUGE-1 Average_F:")+26]
    r2_f = scores_str[scores_str.find("ROUGE-2 Average_F:")+19:scores_str.find("ROUGE-2 Average_F:")+26]
    rl_f = scores_str[scores_str.find("ROUGE-L Average_F:")+19:scores_str.find("ROUGE-L Average_F:")+26]

    return r1_f, r2_f, rl_f


def rouge_file_list(refs_file, hyps_list):
    refs = open(refs_file).readlines()
    scores = rouge_lists(refs, hyps_list)

    return scores


def pprint_rouge_scores(scores, pivot=False):
    pdt = pandas.DataFrame(scores)

    if pivot:
        pdt = pdt.T

    table = tabulate(pdt,
                     headers='keys',
                     floatfmt=".4f", tablefmt="psql")

    return table
