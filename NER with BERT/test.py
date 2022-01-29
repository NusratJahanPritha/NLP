import numpy as np
import torch
from tqdm import trange
from data import device, valid_dataloader, tag_values
from seqeval.metrics import f1_score, accuracy_score
from model import model
from train import validation_loss_values, epochs

# /|\==>VALIDATION(ONEPASS)<==\|/
for _ in trange(epochs, desc="Epoch"):
    model.eval()

    eval_loss, eval_accuracy = 0, 0
    nb_eval_steps, nb_eval_examples = 0, 0
    predictions, true_labels = [], []

    for batch in valid_dataloader:
        batch = tuple(t.to(device) for t in batch)
        b_input_ids, b_input_mask, b_labels = batch

        with torch.no_grad():
            outputs = model(b_input_ids, token_type_ids=None,
                            attention_mask=b_input_mask, labels=b_labels)

        logits = outputs[1].detach().cpu().numpy()
        label_ids = b_labels.to('cpu').numpy()

        eval_loss += outputs[0].mean().item()
        predictions.extend([list(p) for p in np.argmax(logits, axis=2)])
        true_labels.extend(label_ids)
    eval_loss = eval_loss / len(valid_dataloader)
    validation_loss_values.append(eval_loss)
    print('Validation loss: {}'.format(eval_loss))

    pred_tags = [tag_values[p_i] for p, l in zip(predictions, true_labels)
                 for p_i, l_i, in zip(p, l) if tag_values[l_i] != 'PAD']

    valid_tags = [tag_values[l_i] for l in true_labels
                  for l_i in l if tag_values[l_i] != 'PAD']

    print('Validation Accuracy: {}'.format(accuracy_score(pred_tags, valid_tags)))
    print('Validation F-1 Score:{}'.format(f1_score([pred_tags], [valid_tags])))
