import torch
from transformers import *
tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased') 
model = BertModel.from_pretrained('bert-base-multilingual-cased', output_hidden_states=True)


import glob

dir_names = ["DE", "CZ", "IT"]

maxlen = 400
nr_sents_clip = 0

with open("../bert_last_layer_CLS_token.out", "w") as fw:
    with torch.no_grad(): #remove gradient computation
        for lang_name in dir_names:
            data_dir = "../Datasets/"+lang_name
            for fname in glob.glob(data_dir+"/*txt"):
                print(fname)
                text = open(fname, "r").read()
#                vec = tokenizer.encode(text, add_special_tokens=True)
                vec = tokenizer.encode_plus(text, add_special_tokens=True, max_length=maxlen)["input_ids"]
                if len(vec)  == maxlen: nr_sents_clip += 1
#                vec1 = vec[0:maxlen-1]+vec[-1:]
                text_tensor = torch.tensor([vec])
                text_vec = model(text_tensor)
                cls_vec = text_vec[0][0][0].detach().numpy()
                cls_vec = list(map(str, cls_vec))
                #ave_vec = text_vec[0][0][1:-1].numpy().mean(axis=0)
#                ave_vec = list(map(str, ave_vec))
#                if text_vec[0][0].numpy().shape[0] == maxlen: nr_sents_clip += 1

                print(fname, ",".join(cls_vec), sep="\t", file=fw)

print("Nr. clipped sentences {}".format(nr_sents_clip))                
