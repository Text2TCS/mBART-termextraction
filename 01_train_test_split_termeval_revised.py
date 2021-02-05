import os

# ACL2 data augmentation?
acl2_data = False
# What separator to use?
with_underscore = False
with_tag = False
with_comma = True
if with_underscore:
    ext = "_underscore"
elif with_tag:
    ext = "_tag"
elif with_comma:
    ext = "_comma"
else:
    ext = ""

languages = ["en", "fr", "nl", "multi"]
for language in languages:
    if language == "multi":
        langs = ["en", "nl", "fr"]
    else:
        langs = [f"{language}"]


    # Define out dir
    out_dir = f"./preprocessed/{language}{ext}/"
    os.makedirs(out_dir, exist_ok=True)


    # Training Corpus
    # ACL2 - All annotations
    if acl2_data:
        acl2_train_src = open("./acl2/src.txt", "r", encoding="utf-8").readlines()
        acl2_train_label = open("./acl2/label.txt", "r", encoding="utf-8").readlines()
        if with_underscore:
            acl2_train_label = [x.replace(" ", "_") for x in acl2_train_label]
        elif with_tag:
            acl2_train_label = [x.replace("\t", " <eot> ") for x in acl2_train_label]
        elif with_comma:
            acl2_train_label = [x.replace("\t", " ; ") for x in acl2_train_label]
    else:
        acl2_train_src = []
        acl2_train_label = []

    # Training data
    termeval_train_src = []
    termeval_train_label = []
    domains = ["corp", "wind"]
    val_domains = ["equi"]
    for lang in langs:
        for domain in domains:
            termeval_train_src.extend(open(f"./termeval/{lang}_ann/{domain}annfull.txt", "r", encoding="utf-8").readlines())
            termeval_train_label.extend(open(f"./termeval/{lang}_ann/{domain}annlabel.txt", "r", encoding="utf-8").readlines())
    if with_underscore:
        termeval_train_label = [x.replace(" ", "_") for x in termeval_train_label]
    elif with_tag:
        termeval_train_label = [x.replace("\t", " <eot> ") for x in termeval_train_label]
    elif with_comma:
        termeval_train_label = [x.replace("\t", " ; ") for x in termeval_train_label]

    if acl2_data:
        train_src = termeval_train_src
        train_label = termeval_train_label
    else:
        train_src = acl2_train_src + termeval_train_src
        train_label = acl2_train_label + termeval_train_label

    out_train_src = open(f"{out_dir}/train.src", "w", encoding="utf-8")
    out_train_label = open(f"{out_dir}/train.label", "w", encoding="utf-8")
    discarded_lines = 0
    for src, label in zip(train_src, train_label):
        if len(src) > 2:
            out_train_src.write(src)
            out_train_label.write(label)
        else:
            discarded_lines += 1

    print("Lines in train source: ", len(train_src) - discarded_lines)
    print("Lines in train label: ", len(train_label) - discarded_lines)

    # Validation data
    termeval_val_src = []
    termeval_val_label = []
    for lang in langs:
        for domain in val_domains:
            termeval_val_src.extend(open(f"./termeval/{lang}_ann/{domain}annfull.txt", "r", encoding="utf-8").readlines())
            termeval_val_label.extend(open(f"./termeval/{lang}_ann/{domain}annlabel.txt", "r", encoding="utf-8").readlines())
    if with_underscore:
        termeval_val_label = [x.replace(" ", "_") for x in termeval_val_label]
    elif with_tag:
        termeval_val_label = [x.replace("\t", " <eot> ") for x in termeval_val_label]
    elif with_comma:
        termeval_val_label = [x.replace("\t", " ; ") for x in termeval_val_label]
    discarded_lines = 0
    out_val_src = open(f"{out_dir}/valid.src", 'w', encoding='utf-8')
    out_val_label = open(f"{out_dir}/valid.label", 'w', encoding='utf-8')
    for src, label in zip(termeval_val_src, termeval_val_label):
        if len(src) > 2:
            out_val_src.write(src)
            out_val_label.write(label)
        else:
            discarded_lines += 1

    print("Lines in val source: ", len(termeval_val_src) - discarded_lines)
    print("Lines in val label: ", len(termeval_val_label) - discarded_lines)


    # Test/Eval corpus
    termeval_test_src = []
    termeval_test_label = []
    for lang in langs:
        termeval_test_src.extend(open(f"./termeval/{lang}_ann/htflannfull.txt", "r", encoding="utf-8").readlines())
        termeval_test_label.extend(open(f"./termeval/{lang}_ann/htflannlabel.txt", "r", encoding="utf-8").readlines())
    if with_underscore:
        termeval_test_label = [x.replace(" ", "_") for x in termeval_test_label]
    elif with_tag:
        termeval_test_label = [x.replace("\t", " <eot> ") for x in termeval_test_label]
    elif with_comma:
        termeval_test_label = [x.replace("\t", " ; ") for x in termeval_test_label]


    out_test_src = open(f"{out_dir}/test.src", 'w', encoding='utf-8')
    out_test_label = open(f"{out_dir}/test.label", 'w', encoding='utf-8')

    for count, line in enumerate(termeval_test_src):
        if len(line) > 2:
            out_test_src.write(line)
            out_test_label.write(termeval_test_label[count])
    out_test_src.close()
    out_test_label.close()
