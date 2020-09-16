import os
import torch
import sys
from tqdm import tqdm
from torchtext.data.metrics import bleu_score
from nltk.translate.bleu_score import corpus_bleu

# setting device on GPU if available, else CPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def evaluate_subset(test_set, transformer, idx_2_word):
    
    score = 0

    with torch.no_grad():
        transformer.eval()

        final_predictions = []
        final_targets = []

        for message, reply in tqdm(test_set):
            prediction = get_sentence_prediction(transformer, message, idx_2_word)
            reply = [idx_2_word[idx] if idx in idx_2_word else "<UNK>" for idx in reply.view(-1).cpu().numpy()[1:-1]]
            final_predictions.append(prediction[1:-1])
            final_targets.append(reply)


    transformer.train()

    return corpus_bleu(final_targets, final_predictions)
    
def get_sentence_prediction(transformer, message, idx_2_word):
    predictions = [1]
    last_predicted = 1
    idx = 0

    while last_predicted != 2 and idx < message.shape[1]:
        target = torch.LongTensor(predictions).view(1, -1).to(device)
        output = torch.softmax(transformer(message, target), dim=2)
        last_predicted = torch.argmax(output[0][idx])
        predictions.append(last_predicted.item())
        idx += 1

    predictions = [idx_2_word[idx] if idx in idx_2_word else "<UNK>" for idx in predictions]

    return predictions


def test_sentence(transformer, message, idx_2_word, word_2_idx):
    
    message = [word_2_idx[word] for word in message]
    message = torch.LongTensor(message).view(1, -1).to(device)

    return get_sentence_prediction(transformer, message, idx_2_word)

