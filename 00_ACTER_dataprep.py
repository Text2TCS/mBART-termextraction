# -*- coding: utf-8 -*-
import glob
import os
import nltk
import string
import nltk.data
from sacremoses import MosesTokenizer
import re
from tqdm import tqdm
from flashtext.keyword import KeywordProcessor
import ast
from stopwordsiso import stopwords
import shutil


#Prep nltk.punkt models
nltk.download("punkt")

#Define workdir
workdrive = "/mnt/d"
ACTER = "/path/to/ACTER-dataset/" # Path to the ACTER Dataset
workdir = f"./workdir"
outdir = f"./termeval"

### Only annotated or all files?
with_unannotated = False

### Select Language, Corpus and texts
### All texts
if with_unannotated:
    langs = ["en", "fr", "nl"]
    corpora = ["corpall", "equiall", "htflall", "windall"]
    suffix1 = "_unann"
    suffix2 = "all"
else:
### Annotated texts only
    langs = ["en", "fr", "nl"]
    corpora = ["corpann", "equiann", "htflann", "windann"]
    suffix1 = "_ann"
    suffix2 = "ann"

### Copy files to workdir
for lang in langs:
    for corpus in corpora:
        sourcedir = f"{ACTER}/{lang}/{corpus.replace('ann','')}/texts"
        #Annotated files
        os.makedirs(f"{workdir}/{lang}/{corpus}", exist_ok=True)
        os.makedirs(f"{workdir}/{lang}/{corpus.replace('ann', 'all')}", exist_ok=True)
        for root, dirs, files in os.walk(f"{sourcedir}/annotated"):
            for file in files:
                shutil.copy2(f"{os.path.join(root, file)}", f"{workdir}/{lang}/{corpus}/")
        #Annotated + Unannotated files
        if with_unannotated:
        	for root, dirs, files in os.walk(f"{sourcedir}/annotated"):
            	for file in files:
                	shutil.copy2(f"{os.path.join(root, file)}", f"{workdir}/{lang}/{corpus.replace('ann','all')}/")
            	for root, dirs, files in os.walk(f"{sourcedir}/unannotated"):
               	 for file in files:
               	     shutil.copy2(f"{os.path.join(root, file)}", f"{workdir}/{lang}/{corpus.replace('ann', 'all')}/")

### Sentence-split files and concatenate per domain
for lang in langs:
    mt = MosesTokenizer(lang=f'{lang}')
    for corpus in corpora:
        file_list = glob.glob(os.path.join(os.getcwd(), f"{workdir}/{lang}/{corpus}", "*.txt"))

        text =[]

        for file_path in file_list:
            with open(file_path, "r", encoding="utf-8") as f:
                        text.extend(f.read().split("\n"))


        if lang == "en":
            text_sent_tokenized = nltk.sent_tokenize(" ".join(text), language="english")
        elif lang == "fr":
            nltk.data.load('tokenizers/punkt/french.pickle')
            text_sent_tokenized = nltk.sent_tokenize(" ".join(text), language="french")
        elif lang == "nl":
            nltk.data.load('tokenizers/punkt/dutch.pickle')
            text_sent_tokenized = nltk.sent_tokenize(" ".join(text), language="dutch")
        print(text_sent_tokenized)
        text = []
        text_tokenized = []
        for line in text_sent_tokenized:
            text.append(line)
            text_tokenized.append(mt.tokenize(line, return_str=False))

        if with_unannotated:
            out_filename = f"{outdir}/{lang}_unann/{corpus}full.txt"
            out_filename_tok =f"{outdir}/{lang}_unann/{corpus}full_tok.txt"
        else:
            out_filename = f"{outdir}/{lang}_ann/{corpus}full.txt"
            out_filename_tok = f"{outdir}/{lang}_ann/{corpus}full_tok.txt"

# Create Dir if not available and write to file
        os.makedirs(os.path.dirname(out_filename), exist_ok=True)
        out = open(out_filename,"w", encoding="utf-8")
        out_tok = open(out_filename_tok, "w", encoding="utf-8")
        for t in text:
            if re.match(r'^\s*$', "".join(t)):
                continue
            else:
                out.write("".join(t) + "\n")
        for t in text_tokenized:
            if re.match(r'^\s*$', "".join(t)):
                continue
            else:
                out_tok.write(" ".join(t) + "\n")

        out.close()
        out_tok.close()

### Perpare termextraction

# KeywordProcessor Preparation
kwp = KeywordProcessor(case_sensitive=False)

# Kwp only treats "\w" and [a-zA-Z0-9] as word boundaries, add whatever else seems important
dict_file = open(f"{workdrive}/coding/Text2TCS/scripts/textprocessing/wordboundaries/accents.txt", "r",
                 encoding='utf-8')
word_boundary_dict = ast.literal_eval(dict_file.read())
word_boundary_list = ["-", "—", "/"]
for i in word_boundary_dict.keys():
    word_boundary_list.append(i)
cyrillic = [i for i in "АаБбВвГгДдЕеЁёЖжЗзИиЙйКкЛлМмНнОоПпРрСсТтУуФфХхЦцЧчШшЩщЪъЫыЬьЭэЮюЯя"]
greek = [i for i in "ΑΒΓΔΕΖΗΘΙΚΛΜΝΞΟΠΡΣΤΥΦΧΨΩαβγδεζηθικλμνξοπρστυφχψω"]
word_boundary_list.extend(cyrillic)
word_boundary_list.extend(greek)
for char in word_boundary_list:
    kwp.add_non_word_boundary(char)

# Automate language and domain
langs = ["en", "fr", "nl"]
domains = ["corp", "equi", "htfl", "wind"]

for lang in langs:
    mt = MosesTokenizer(lang=f'{lang}')
    for domain in domains:
        sourcedir = f"{workdrive}/coding/Text2TCS/git/ACTER/{lang}/{domain}/annotations/"
        list_of_terms = []
        with open(f"{sourcedir}/{domain}_{lang}_terms_nes.ann",
                  "r", encoding="utf-8") as f:
            for line in f.readlines():
                s = str(line).replace("OOD_Term", "").replace("Common_Term", "").replace("Specific_Term", "").replace(
                    "Named_Entity", "").strip("\n").strip("\t")
                list_of_terms.append(s)

        kwp.add_keywords_from_list(list_of_terms)

        # Remove unwanted terms from list (single letters, prepositions, stop-words etc.)
        abc_list = list(string.ascii_uppercase + string.ascii_lowercase)
        kwp.remove_keywords_from_list(abc_list)
        kwp.remove_keywords_from_list(word_boundary_list)

        for i in stopwords(f"{lang}"):
            kwp.remove_keyword(i)
            kwp.remove_keyword(i.capitalize())

        # Extract the terms
        with open(f"{outdir}/{lang}{suffix1}/{domain}{suffix2}full_tok.txt", "r", encoding="utf-8") as f:
            sentences = f.readlines()
#        print(sentences[-10:])
        results =[]
        for line in tqdm(sentences):
            s = kwp.extract_keywords(line.rstrip())
            results.append(s)
        print(results[-10:])

        # Remove previously added terms from keyword processor
        for i in list_of_terms:
            kwp.remove_keyword(" ".join(i))

        # Write results:
        out = open(f"{outdir}/{lang}{suffix1}/{domain}{suffix2}label.txt","w",encoding="utf-8")

        for s in results:
            if re.match(r'^\s*$', "".join(s)):
                out.write("" + "\n")
            else:
                out.write("\t".join(s) + "\n")
        out.close()


