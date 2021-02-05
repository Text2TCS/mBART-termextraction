### Scikit F1 Score
import os
import re
from collections import Counter

def flatten(l):
    return [item for sublist in l for item in sublist]


def computeTermEvalMetrics(extracted_terms, gold_df):
  #make lower case cause gold standard is lower case
  extracted_terms = set([item.lower() for item in extracted_terms])
  gold_set=set(gold_df)
  true_pos=extracted_terms.intersection(gold_set)
  false_pos=extracted_terms - true_pos
  false_neg=gold_set - extracted_terms
  recall=len(true_pos)/len(gold_set)
  precision=len(true_pos)/len(extracted_terms)
  d = dict()
  d["Intersection"] = len(true_pos)
  d["Gold"] = len(gold_set)
  d["Extracted"] = len(extracted_terms)
  d["False Positive"] = len(false_pos)
  d["False Negative"] = len(false_neg)
  d["Precision:"] = precision*100
  d["Recall:"] = recall*100
  d["F1:"] = 2*(precision*recall)/(precision+recall)*100
  return d, false_pos, false_neg, true_pos

# Should output be written to file?
write_to_file = True
show_termtypes = False

# Use comma hack?
comma_hack = True

# Define workdirs

ACTER = "/path/to/ACTER-dataset/"
workdir = "."
inputdir = f"{workdir}/out"
print(inputdir)

os.system(f"for l in {inputdir}/*; do grep ^D $l/generate-test.txt | cut -f3- > $l/out.sys; done")

avg_termtype_count = Counter()
avg_termtype_count_gold = Counter()
for root, dirs, files in os.walk(inputdir):
   for directory in dirs:
        sys = f"{os.path.join(root, directory)}/out.sys"
        if re.search('_underscore.+', directory):
            system = [x.replace("_ ", "_").split() for x in open(sys, "r", encoding="utf-8").readlines()]
        elif re.search('_tag.+', directory):
            system = [x.split(' <eot>') for x in open(sys, "r", encoding="utf-8").readlines()]
        elif re.search('_comma.+', directory) and comma_hack:
            system = [x.split(' ;') for x in open(sys, "r", encoding="utf-8").readlines()]
        elif re.search('_comma.+', directory) and not comma_hack:
            system = [x.split(' ; ') for x in open(sys, "r", encoding="utf-8").readlines()]
        else:
            print(f"Skipping {directory}...")
            continue


        eval_langs = ['en', 'fr', 'nl']
        for lang in eval_langs:
            gold_train = []
            gold_train_ann = {}
            for domain in ["corp", "wind"]:
                termeval_train_ref = f"{ACTER}/{lang}/{domain}/annotations/{domain}_{lang}_terms_nes.ann"
                with open(termeval_train_ref, "r", encoding="utf-8") as f:
                    for line in f.readlines():
                        if re.search('.+underscore', directory):
                            s = line.replace(" ", "_").strip("\n")
                        else:
                            s = line.strip("\n")
                        gold_train.append(s.split("\t")[0])
                        gold_train_ann[s.split('\t')[0]] = s.split('\t')[1]

            if re.search(fr'.+_{lang}', directory):
                termeval_ref = f"{ACTER}/htfl/annotations/htfl_{lang}_terms_nes.ann"
                gold = []
                gold_ann = {}
                with open(termeval_ref, "r", encoding="utf-8") as f:
                    for line in f.readlines():
                        if re.search('.+underscore', directory):
                            s = line.replace(" ", "_").strip("\n")
                        else:
                            s = line.strip("\n")
                        gold.append(s.split("\t")[0]) #Gold without annotation
                        gold_ann[s.split('\t')[0]] = s.split('\t')[1]

                termlist = []
                for line in system:
                    for word in line:
                        termlist.append(word.lower().strip())
                extracted_terms = set(termlist)

                if write_to_file:
                    with open(f"{inputdir}/{directory}/extracted_terms.txt", "w") as f:
                        for line in sorted(extracted_terms,key=str.lower):
                            if len(line) > 0:
                                print(line, file=f)

#F1-Score and reporting:

                print(f"{directory} :")
                score, false_pos, false_neg, true_pos = computeTermEvalMetrics(extracted_terms, gold)
                print(round(score['Precision:'], 1), "/", round(score['Recall:'], 1), "/", round(score["F1:"], 1))
                print("False Positive: ", score['False Positive'], "False Negative: ", score['False Negative'])

                # Proportion of Specific, Common and OOD
                if show_termtypes:
                    if write_to_file:
                        out_termtype = open(f"{inputdir}/{directory}/termtypes_mBART_{directory}.txt", "w")
                    for index, dataset in enumerate([set(gold), true_pos]):
                        output_list = ["GOLD", "TRUE_POS"]
                        termtype_list = [gold_ann[term] for term in dataset]
                        termtype_count = {termtype: termtype_list.count(termtype) for termtype in set(termtype_list)}
                        print(output_list[index], ": ", termtype_count)
                        if write_to_file:
                            if index == 0:
                                print(f"Term types in {directory}", "\n", file=out_termtype)
                            print(output_list[index], "\n", termtype_count, "\n", file=out_termtype)
                        if re.search('multi_comma.+', directory) and index == 0:
                            avg_termtype_count_gold.update(termtype_count)
                        if re.search('multi_comma.+', directory) and index == 1:
                            avg_termtype_count.update(termtype_count)
                    if write_to_file:
                        out_termtype.close()

                # Termlenght:
                if write_to_file:
                    out_termlenght = open(f"{inputdir}/{directory}/termlenght_mBART_{directory}.txt", "w")
                for index, terms in enumerate([false_pos, false_neg, gold, gold_train, extracted_terms]):
                    output_list= ["FP", "FN", "GOLD", "TRAIN_GOLD", "EXTRACTED"]
                    if re.search('_underscore.+', directory):
                        termlength = [len(term.split("_")) for term in terms]
                    else:
                        termlength = [len(term.split(" ")) for term in terms]
                    termcount = []
                    for word_count in set(termlength):
                        if round(termlength.count(word_count) / len(terms), 3) > 0.001:
                            termcount.append([word_count, round(termlength.count(word_count) / len(terms), 2)])
                    #print(output_list[index], termcount)
                    if write_to_file:
                        if index == 0:
                            print(f"Term lengths in {directory}", "\n", file=out_termlenght)
                        print(output_list[index], "\n", termcount, "\n", file=out_termlenght)
                if write_to_file:
                    out_termlenght.close()

                if write_to_file:
                    out_fpos = open(f"{inputdir}/{directory}/false_positives_mBART_{directory}.txt", "w")
                    out_fneg = open(f"{inputdir}/{directory}/false_negatives_mBART_{directory}.txt", "w")
                    for fpos in sorted(false_pos, key=str.lower):
                            if len(fpos) > 0:
                                out_fpos.write(fpos + "\n")
                    for fneg in sorted(false_neg, key=str.lower):
                            out_fneg.write(fneg + "\n")
                    out_fpos.close()
                    out_fneg.close()

# Write termtypes
if write_to_file and show_termtypes:
    with open (f"{inputdir}/best_model_termtypes.txt", "w") as f:
        print('GOLD TERMTYPES ACROSS ALL LANGUAGES:')
        print('GOLD TERMTYPES ACROSS ALL LANGUAGES:', file=f)
        total_gold = 0
        for key,value in avg_termtype_count_gold.items():
            total_gold += value / 3
        for key, value in avg_termtype_count_gold.items():
            print(key, round(value / 3, 3), "(", round((value / 3) / total_gold * 100, 3), "%)")
            print(key, round(value / 3, 3), "(",round((value / 3) / total_gold * 100, 3), "%)", file=f)
        print('BEST MODEL TERMTYPES ACROSS ALL LANGUAGES:')
        print('\nBEST MODEL TERMTYPES ACROSS ALL LANGUAGES:', file=f)
        total = 0
        for key, value in avg_termtype_count.items():
            total += value / 3
        for key, value in avg_termtype_count.items():
            print(key, round(value / 3, 3), "(", round((value / 3) / total * 100, 3), "% )")
            print(key, round(value / 3, 3), "(", round((value / 3) / total * 100, 3), "% )", file=f)
