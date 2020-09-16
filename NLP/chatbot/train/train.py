import torch.nn as nn
import os
import torch
from tqdm.auto import tqdm
import sys
sys.path.insert(1, r"D:\Projects\NLP\chatbot\util")
sys.path.insert(2, r"D:\Projects\NLP\chatbot\model")
sys.path.insert(2, r"D:\Projects\NLP\chatbot\eval")
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from dataset_loader import DatasetLoader, my_collate, PATH
from transformer import Transformer
from eval import evaluate_subset, test_sentence

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

torch.cuda.empty_cache()
loader = DatasetLoader(PATH)
train_subset, test_subset = torch.utils.data.random_split(loader, [int(0.999 * loader.__len__() + 1), int(0.001 * loader.__len__())])
trainset = DataLoader(dataset = train_subset,
                        batch_size = 30,
                        shuffle = True,
                        collate_fn = my_collate)

testset = DataLoader(dataset = test_subset,
                     batch_size = 1,
                     shuffle = True)

VOCAB_SIZE = loader.__len__()
PAD_IDX = 0
EMBEDDING_SIZE = 512
HEADS = 8
SAVE_PATH = r"D:\Projects\NLP\chatbot\model\chatbot.pth"

transformer = Transformer(VOCAB_SIZE, VOCAB_SIZE, PAD_IDX, PAD_IDX).to(device)
transformer.load_state_dict(torch.load(SAVE_PATH))
optimizer = torch.optim.Adam(transformer.parameters(), lr=3e-4)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, factor=0.1, patience=10, verbose=True
)
criterion = nn.CrossEntropyLoss(ignore_index=0)

test_sentence_1 = ["<SOS>", "do", "you", "like", "horses", "<EOS>"]
test_sentence_2 = ["<SOS>", "are", "you", "alive", "<EOS>"]

def train():
    loss = 0
    epoch = 0 
    while epoch < 10000:
        loss = 0
        for message, reply in tqdm(trainset):
            optimizer.zero_grad()
            out = transformer(message, reply[:, :-1])
            out = out.reshape(-1, out.shape[2])
            reply = reply[:, 1:].reshape(-1)

            loss_it = criterion(out, reply)
            loss_it.backward()
            loss += loss_it.item()

            torch.nn.utils.clip_grad_norm_(transformer.parameters(), max_norm=1)
            optimizer.step()

        torch.save(transformer.state_dict(), SAVE_PATH)
        mean_loss = loss / (trainset.__len__() * 30)
        scheduler.step(mean_loss)
        print("Bleu score for epoch {} is {}".format(epoch, evaluate_subset(testset, transformer, loader.idx_2_word)))
        print("Loss for epoch {} is {}".format(epoch, loss))
        print(test_sentence_1)
        print(test_sentence(transformer, test_sentence_1, loader.idx_2_word, loader.word_2_idx))
        print('------')
        print(test_sentence_2)
        print(test_sentence(transformer, test_sentence_2, loader.idx_2_word, loader.word_2_idx))

train()
#print(evaluate_subset(testset, transformer, loader.idx_2_word))