import pandas as pd
import torch
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from transformers import BertTokenizer
from sklearn.model_selection import train_test_split
from keras_preprocessing.sequence import pad_sequences

MAX_LEN = 75
BATCH_SIZE = 16

data = pd.read_csv('ner_datasetreference.csv', encoding='latin1')
data = data.fillna(method='ffill')
entities_to_remove = ["B-gpe", "I-gpe", "B-tim", "I-tim", "B-art", "I-art", "B-eve", "I-eve", "B-nat", "I-nat"]
data = data[~data.Tag.isin(entities_to_remove)]
data = data.drop(['POS'], axis=1)
data.head()


class GetSentence(object):
    def __init__(self, data):
        self.data = data
        self.n_sentences = 1
        self.empty = False
        agg_function = lambda s: [(w, t) for w, t in zip(s["Word"].values.tolist(),
                                                         s["Tag"].values.tolist())]
        self.group = self.data.groupby('Sentence #').apply(agg_function)
        self.sentence = [s for s in self.group]


getter = GetSentence(data)

sentence = [[word[0] for word in sentence] for sentence in getter.sentence]
labels = [[lab[1] for lab in sentence] for sentence in getter.sentence]
tag_values = list(set(data["Tag"].values))
tag_values.append('PAD')
tag_idx = {t: i for i, t in enumerate(tag_values)}

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

tokenizer = BertTokenizer.from_pretrained('bert-base-cased', do_lower_case=False)


def tokenize_preserve(sentences, text_labels):
    tokenized_sentence = []
    labels = []

    for word, label in zip(sentences, text_labels):
        tokenized_word = tokenizer.tokenize(word)
        n_subwords = len(tokenized_word)
        tokenized_sentence.extend(tokenized_word)
        labels.extend([label] * n_subwords)

    return tokenized_sentence, labels


tokenized_texts_and_labels = [
    tokenize_preserve(sent, labs)
    for sent, labs in zip(sentence, labels)
]

tokenized_text = [token_label_pair[0] for token_label_pair in tokenized_texts_and_labels]
labels = [token_label_pair[1] for token_label_pair in tokenized_texts_and_labels]
len(tokenized_text[0]), len(labels[0])
input_ids = pad_sequences([tokenizer.convert_tokens_to_ids(txt) for txt in tokenized_text],
                          maxlen=MAX_LEN, dtype='long', value=0.0,
                          truncating='post', padding='post')
tags = pad_sequences([[tag_idx.get(l) for l in lab] for lab in labels],
                     maxlen=MAX_LEN, dtype='long', value=tag_idx['PAD'],
                     truncating='post', padding='post')
attention_masks = [[float(i != 0.0) for i in ii] for ii in input_ids]
tr_input, val_input, tr_tag, val_tag = train_test_split(input_ids, tags, random_state=45, test_size=.1)
tr_masks, val_masks, _, _ = train_test_split(attention_masks, input_ids, random_state=45, test_size=.1)
len(tr_masks[0]), len(tr_input[0]), len(tr_tag[0])
tr_input = torch.tensor(tr_input).to(dtype=torch.long)
val_input = torch.tensor(val_input).to(dtype=torch.long)
tr_tag = torch.tensor(tr_tag).to(dtype=torch.long)
val_tag = torch.tensor(val_tag).to(dtype=torch.long)
tr_masks = torch.tensor(tr_masks).to(dtype=torch.long)
val_masks = torch.tensor(val_masks).to(dtype=torch.long)
train_data = TensorDataset(tr_input, tr_masks, tr_tag)
train_sampler = RandomSampler(train_data)
train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=BATCH_SIZE)

valid_data = TensorDataset(tr_input, tr_masks, tr_tag)
valid_sampler = SequentialSampler(valid_data)
valid_dataloader = DataLoader(valid_data, sampler=valid_sampler, batch_size=BATCH_SIZE)
