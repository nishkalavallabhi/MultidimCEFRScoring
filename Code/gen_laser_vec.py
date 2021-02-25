from laserembeddings import Laser

import glob
import numpy as np

laser = Laser()

iso_codes = ["de", "cz", "it"]
dir_names = ["DE", "CZ", "IT"]

with open("../laser_vecs.out", "w") as fw:
    for lang_name in dir_names:
        data_dir = "../Datasets/"+lang_name
        for fname in glob.glob(data_dir+"/*txt"):
            lines = open(fname, "r").readlines()
            iso_code = lang_name.lower()
            vecs = laser.embed_sentences(lines, lang = iso_code)
            ave_vec = vecs.mean(axis=0)
            ave_vec = list(map(str, ave_vec))
            print(fname, vecs.shape)
            print(fname, ",".join(ave_vec), sep="\t", file=fw)
