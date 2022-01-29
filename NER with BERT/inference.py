import numpy as np
import torch
import pandas as pd
import os
import argparser

args = argparser.parser.parse_args()

print(args.input)
input = ""

if str(args.input).__contains__(".txt"):
    filepath = args.input
    with open(filepath) as f:
        inp = f.readlines()
        for line in inp:
            input += line

elif str(args.input).__contains__(".csv"):
    filepath = args.input
    with open(filepath) as f:
        inp = f.readlines()
        for line in inp:
            words = line.split(",")
            for w in words:
                input += w + " "

elif str(args.input).__contains__(".tsv"):
    filepath = args.input
    with open(filepath) as f:
        inp = f.readlines()
        for line in inp:
            words = line.split("\t")
            for w in words:
                input += w + " "

else:
    raise Exception("Input file has to be of .txt or .csv or .tsv type")

model = torch.load("model.pt", map_location='cpu')
model.eval()

tokenized_sentence = config.tokenizer.encode(input)
input_ids = torch.tensor([tokenized_sentence])

with torch.no_grad():
    output = model(input_ids)
    label_indices = np.argmax(output[0].to('cpu').numpy(), axis=2)

tokens = config.tokenizer.convert_ids_to_tokens(input_ids.to('cpu').numpy()[0])

new_tokens, new_labels = [], []

for token, label_idx in zip(tokens, label_indices[0]):
    if token.startswith('##'):
        new_tokens[-1] = new_tokens[-1] + token[2:]
    else:
        new_labels.append(config.tag_values[label_idx])
        new_tokens.append(token)

names = []
locs = []
orgs = []

curr_name = ""
curr_loc = ""
curr_org = ""
for i in range(len(new_tokens)):
    if new_labels[i] != 'O':
        leb = new_labels[i]
        if leb == "B-per":
            if curr_loc != "":
                locs.append(curr_loc)
                curr_loc = ""
            if curr_org != "":
                orgs.append(curr_org)
                curr_org = ""
            if curr_name != "":
                curr_name += " "
            curr_name += new_tokens[i]

        if leb == "I-per":
            if curr_loc != "":
                locs.append(curr_loc)
                curr_loc = ""
            if curr_org != "":
                orgs.append(curr_org)
                curr_org = ""
            if curr_name != "":
                curr_name += " "
            curr_name += new_tokens[i]

        if leb == "B-geo":
            if curr_name != "":
                names.append(curr_name)
                curr_name = ""
            if curr_org != "":
                orgs.append(curr_org)
                curr_org = ""
            if curr_loc != "":
                curr_loc += " "
            curr_loc += new_tokens[i]

        if leb == "I-geo":
            if curr_name != "":
                names.append(curr_name)
                curr_name = ""
            if curr_org != "":
                orgs.append(curr_org)
                curr_org = ""
            if curr_loc != "":
                curr_loc += " "
            curr_loc += new_tokens[i]

        if leb == "B-org":
            if curr_loc != "":
                locs.append(curr_loc)
                curr_loc = ""
            if curr_name != "":
                names.append(curr_name)
                curr_name = ""
            if curr_org != "":
                curr_org += " "
            curr_org += new_tokens[i]

        if leb == "I-org":
            if curr_loc != "":
                locs.append(curr_loc)
                curr_loc = ""
            if curr_name != "":
                names.append(curr_name)
                curr_name = ""
            if curr_org != "":
                curr_org += " "
            curr_org += new_tokens[i]


    else:
        if curr_name != "":
            names.append(curr_name)
            curr_name = ""
        if curr_loc != "":
            locs.append(curr_loc)
            curr_loc = ""
        if curr_org != "":
            orgs.append(curr_org)
            curr_org = ""

df = pd.DataFrame()
df["Free flow of Text"] = [input]
df["Extracted Name"] = [names]
df["Extracted Location"] = [locs]
df["Extracted Organization"] = [orgs]

print(df)

df.to_csv("output1.csv", index=False)