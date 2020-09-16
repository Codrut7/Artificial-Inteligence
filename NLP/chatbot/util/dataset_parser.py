import os
import string
import unidecode
import unicodedata
import re

# Ignore warnings
import warnings
warnings.filterwarnings("ignore")

class DatasetParser():

    def __init__(self, dataset_path, target_path):
        """
        Args :
            dataset_path : The path to the dataset.
            target_path : The path for the parsed dataset.
        """

        # Root directory / Target directory
        self.dataset_path = dataset_path
        self.target_path = target_path
        # List of sentences
        self.dataset = self.load_data(self.dataset_path)
        # Input/ Target conversation from the dataset
        self.input, self.target = self.parse_dataset(self.dataset)
        # Write the parsed new file for 
        self.write_file(self.target_path)

    # Turn a Unicode string to plain ASCII, thanks to
    # https://stackoverflow.com/a/518232/2809427
    def unicodeToAscii(self, s):
        return ''.join(
            c for c in unicodedata.normalize('NFD', s)
            if unicodedata.category(c) != 'Mn'
        )

    def load_data(self, dataset_path):
        """
        Args :
            dataset_path : The path to the dataset json from the dataset
        Returns : 
            dataset list with the lines from the subtitle files
        """
        dataset = []

        for subtitle in os.listdir(dataset_path):
            subtitle_path = os.path.join(dataset_path, subtitle)

            if subtitle_path.endswith('.json'): # if the data is from the MULTIWOZ dataset
                with open(subtitle_path, encoding='utf8') as f:
                    lines = f.readlines()
                lines = [line for line in lines if "text" in line]
                lines = [line.rstrip().lstrip().replace('"text":', '') for line in lines] 
            else: # if the dataset is from Connel line movies
                with open(subtitle_path, encoding='iso-8859-1') as f:
                    lines = f.readlines()
                lines = [line.split('+++$+++')[len(line.split('+++$+++'))-1] for line in lines] # take only the sentence from the dataset
                lines = [line.rstrip().lstrip() for line in lines]

            lines = [line.replace('\n', '') for line in lines]   # remove the endline
            dataset += lines

        return dataset

    def parse_dataset(self, dataset):
        """
        Args :
            dataset : the list of lines from the subtitles
        Returns :
            input : the input sentence of one character 
            target : the target response for the input sentence
        """
        
        inp = []
        target = []

        for idx in range(0, len(dataset)-1, 2): # parse 2 sentences at a time
            inp.append(self.clear_sentence(dataset[idx]))
            target.append(self.clear_sentence(dataset[idx+1]))
        
        return inp, target

    def clear_sentence(self, sentence):
        """
        Args:
            sentence : The sentence to be cleared.
        """
        s = self.unicodeToAscii(sentence.lower().strip())
        s = re.sub(r"([.!])", r"", s)
        s = re.sub(r"\s+", r" ", s).strip()
        return s

    def write_file(self, target_path):
        with open(os.path.join(target_path, "dataset.txt"), "w",  encoding="utf-8") as f:
            for i in range(len(self.input)):
                if self.input[i] == '' or self.target == '':
                    continue
                f.write(self.input[i] + '\t' + self.target[i] + '\n')
            

DATASET_PATH = r"C:\Projects\NLP\chatbot\subtitrari"
WRITE_PATH = r"C:\Projects\NLP\chatbot\subtitrari"

parser = DatasetParser(DATASET_PATH, WRITE_PATH)