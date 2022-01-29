import numpy as np
import torch
from torch import nn
from tqdm import trange
from data import train_dataloader, device, valid_dataloader, tag_values
from seqeval.metrics import f1_score, accuracy_score
from model import model, optimizer, scheduler
import matplotlib.pyplot as plt
import seaborn as sns


epochs = 3
max_grad_norm = 1.0
# storing loss values
# storing loss values
loss_values, validation_loss_values = [], []

# TRAINING AND VALIDATION

for _ in trange(epochs, desc="Epoch"):
    # /|\==>TRAINLOOP(ONEPASS)<==\|/
    model.train()
    total_loss = 0  # so it resets each epoch

    for step, batch in enumerate(train_dataloader):
        batch = tuple(t.to(device) for t in batch)
        b_input_ids, b_input_mask, b_labels = batch  # also the order in train_data/val_data

        model.zero_grad()  # clearing previous gradients for each epoch

        outputs = model(b_input_ids, token_type_ids=None,
                        attention_mask=b_input_mask, labels=b_labels)  # forward pass

        loss = outputs[0]
        loss.backward()  # getting the loss and performing backward pass

        total_loss += loss.item()  # tracking loss

        torch.nn.utils.clip_grad_norm(parameters=model.parameters(), max_norm=max_grad_norm)
        # ^^^ preventing exploding grads

        optimizer.step()  # updates parameters

        scheduler.step()  # update learning_rate

    avg_train_loss = total_loss / len(train_dataloader)
    print('Average train loss : {}'.format(avg_train_loss))

    loss_values.append(avg_train_loss)  # storing loss values if you choose to plot learning curve

