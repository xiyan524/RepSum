import files2rouge
import chardet
import codecs
import os

dec_path = ""
ref_path = ""
result_id_path = ""


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

def trans_id(dec_path, ref_path):
    dec_files = codecs.open(dec_path, encoding="utf-8").read().split("\n")
    ref_files = codecs.open(ref_path, encoding="utf-8").read().split("\n")

    dec_files_id = codecs.open(os.path.join(result_id_path, "decode_id_tmp.txt"), 'a')
    ref_files_id = codecs.open(os.path.join(result_id_path, "reference_id_tmp.txt"), 'a')

    sample_num = len(dec_files)
    for index in range(sample_num):
        dec_file = dec_files[index].split(" ")
        ref_file = ref_files[index].split(" ")
        dec_id, ref_id = tokens_to_ids(dec_file, ref_file)
        write_id(dec_id, dec_files_id)
        write_id(ref_id, ref_files_id)


#trans_id(dec_path, ref_path)
#files2rouge.run(os.path.join(result_id_path, "decode_id_tmp.txt"), 
#os.path.join(result_id_path, "reference_id_tmp.txt"),
#os.path.join(result_id_path, "results.txt"))

files2rouge.run(ref_path, dec_path)