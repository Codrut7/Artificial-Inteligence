import torch
import os
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence

# setting device on GPU if available, else CPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def my_collate(batch):
    """
    Batch processing. Used in order to pad a batch to the max_len seq size.
    Args :
        batch : tuple input sentence, response sentence.
    """
    inp = [i[0] for i in batch]
    target = [i[1] for i in batch]
    inp = torch.nn.utils.rnn.pad_sequence(inp, batch_first=True)
    target = torch.nn.utils.rnn.pad_sequence(target, batch_first=True)

    return inp.type(torch.LongTensor).to(device), target.type(torch.LongTensor).to(device)

class DatasetLoader(Dataset):

    def __init__(self, dataset_path, MIN_WORD_FREQ=3):
        """
        Args :
            dataset_path : the path to the loaded data set
        """
        self.MIN_WORD_FREQ = MIN_WORD_FREQ
        self.dataset_file = open(dataset_path, 'r', encoding='utf-8')
        self.lines = self.dataset_file.readlines()
        self.input, self.target = self.parse_lines(self.lines)
        self.word_2_frq = self.count_words()
        self.word_2_idx, self.idx_2_word = self.build_dictionary()
        

    def __len__(self):
        return len(self.input)

    def parse_lines(self, lines):
        """
        Args :
            lines : the lines of the dataset
        """
        inp = []
        target = []

        for line in lines:
            try:
                input_line, target_line = line.split('\t')
            except:
                continue
            if input_line == '' or target_line == '':
                continue
            if len(input_line.split(' ')) >= 20 or len(target_line.split(' ')) >= 20:
                continue
            
            input_line = input_line.rstrip().lstrip()
            target_line = target_line.rstrip().lstrip()
            inp.append(input_line.split(' '))
            target.append(target_line.split(' '))
        
        return inp, target

    def count_words(self):
        """
        Create a dictionary with the frequency of each word.
        """
        word_2_frq = {}

        for i in range(len(self.input)):
            # count the frequency of input words
            for word in self.input[i]:
                if word not in word_2_frq:
                    word_2_frq[word] = 1
                else:
                    word_2_frq[word] += 1
            # count the frequency of target words
            for word in self.target[i]:
                if word not in word_2_frq:
                    word_2_frq[word] = 1
                else:
                    word_2_frq[word] += 1

        return word_2_frq

    def build_dictionary(self):
        """
        Transform the input/target sentences to dictionaries.
        """
        word_2_idx = {"<SOS>" : 1, "<EOS>" : 2, "<UNK>" : 3}
        idx_2_word = {1 : "<SOS>", 2 : "<EOS>", 3 : "<UNK>"}

        for i in range(len(self.input)):
            # idx dictionary for the words
            for word in self.input[i]:
                if word not in word_2_idx and self.word_2_frq[word] > self.MIN_WORD_FREQ:
                    word_2_idx[word] = len(word_2_idx)
                    idx_2_word[len(word_2_idx)-1] = word
            # word for each idx
            for word in self.target[i]:
                if word not in word_2_idx and self.word_2_frq[word] > self.MIN_WORD_FREQ:
                    word_2_idx[word] = len(word_2_idx)
                    idx_2_word[len(word_2_idx)-1] = word
        
        return word_2_idx, idx_2_word

    def __getitem__(self, index):
        """
        Extracts a certain line from the dataset of conversations.
        
        Args : 
            index : index of the line to be extracted.
        """
        
        inp, target = self.input[index], self.target[index]
        
        inp =  [self.word_2_idx["<SOS>"]] + [self.word_2_idx[word] if word in self.word_2_idx else self.word_2_idx["<UNK>"] for word in inp] + [self.word_2_idx["<EOS>"]]
        target = [self.word_2_idx["<SOS>"]] + [self.word_2_idx[word] if word in self.word_2_idx else self.word_2_idx["<UNK>"]  for word in target] + [self.word_2_idx["<EOS>"]]

        inp = torch.LongTensor(inp).to(device)
        target = torch.LongTensor(target).to(device)

        return inp, target


PATH = r"D:\Projects\NLP\chatbot\subtitrari\dataset.txt"
#trainset = DataLoader(dataset = DatasetLoader(PATH),
#                        batch_size = 4,
#                        shuffle = True,
#                        collate_fn = my_collate)