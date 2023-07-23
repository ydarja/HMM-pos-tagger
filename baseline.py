"""
 Course:        Statistical Language Processing - Summer 2023
 Assignment:    (A3)
 Author:        (Daniel Stuhlinger, Darja Jepifanpva)

 Honor Code:    We pledge that this program represents our own work.
"""
from tagger import Tagger
import numpy as np
from sklearn.metrics import classification_report


class Baseline(Tagger):
    def __init__(self):
        super().__init__()

        self.word_pos_map = None  # {word: pos} dictionary

    def train_model(self, train_file):
        """Train a baseline model using the CoNLL-U data in the training file.

        Use the load_data() function in the parent class (Tagger) to load the
        training data. The class variables set by Tagger.load_data() are
        available here as well. For example, access the training sentences
        with self.train_sents.

        The baseline implementation simply calculates the most frequent POS tag
        for each word in the training data. For example, if the word "trip" occurs
        as NOUN 100 times, and as VERB 30 times, then the key "trip" in word_pos_map
        will have the value NOUN.

        In case of a tie, choose the tag with the lowest index. Since the tags are
        sorted, this will also be the tag which comes first alphabetically. For
        example, if a word occurs 5 times each as INTJ, PUNCT, SYM (and less than
        5 times for each of the other tags), it is assigned INTJ.

        Set the class variable for the component of the baseline model:
            - word_pos_map

        Parameters
        ----------
        train_file: str
            CoNLL-U training file

        """
        # load the train data
        self.load_data(train_file, train=True)
        
        # flatten the arrays of words and tags
        flattened_sents = [word for sent in self.train_sents for word in sent]
        flattened_tags = [tag for sent in self.train_tags for tag in sent]
        # create a list of tuples (word, pos)
        word_pos = list(zip(flattened_sents, flattened_tags))

        # create and fill in a nested frequency dictionary 
        # of form {word: {pos1: freq1, pos2: freq2, ...}}
        freq_dict = dict()
        for word, pos in word_pos:
            if word not in freq_dict:
                freq_dict[word] = {pos:1}
            else:
                if pos not in freq_dict[word]:
                    freq_dict[word][pos] = 1
                else:
                    freq_dict[word][pos] += 1

        # initialize word_pos_map and choose the most frequent POS tag for each word
        self.word_pos_map = dict()           
        for word in freq_dict:
            freq_dict[word] = dict(sorted(freq_dict[word].items()))
            self.word_pos_map[word] = max(freq_dict[word], key=freq_dict[word].get)
        # add UNK token with alphabetically first POS tag in the set    
        self.word_pos_map["<UNK>"] = sorted(list(self.pos_tagset))[0]
        self.word_pos_map = dict(sorted(self.word_pos_map.items()))

    def decode(self, sentence):
        """Use the simple baseline algorithm to find the POS tagging for the
        input sentence, which is a dictionary look-up for the POS tag for each word.

        Parameters
        ----------
        sentence : list[str]
            The sentence to tag

        Returns
        -------
        pred_tags : list[str]
            predicted POS tags, one tag per word in input sentence
        """

        return [self.word_pos_map[word] for word in sentence]


def main(train_file, test_file):
    """Train and test the baseline tagger.
    Print the evaluation report and the macro f1-score.

    Parameters
    ----------
    train_file : str
        conllu file to use for training
    test_file : str
        conllu file to use for testing
    """
    m1 = Baseline()
    # train the model
    m1.train_model(train_file)
    # evaluate the model
    report_str, f1_score = m1.evaluate_model(test_file)
    print(f'Report:\n{report_str}\nF1 Score = {f1_score}')

if __name__ == '__main__':
    main('Data/en_ewt-ud-train.conllu', 'Data/en_ewt-ud-test.conllu')
    
