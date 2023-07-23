"""
 Course:        Statistical Language Processing - Summer 2023
 Assignment:    (A3)
 Author:        (Daniel Stuhlinger, Darja Jepifanpva)

 Honor Code:    We pledge that this program represents our own work.
"""
import pyconll
from sklearn.metrics import classification_report, f1_score

UNK = "<UNK>"
EOS = '</s>'
END = 'END'


class Tagger:
    def __init__(self):

        self.train_sents = None  # list[list[str]]
        self.train_tags = None  # list[list[str]]
        self.vocab = None  # list[str]
        self.pos_tagset = None  # list[str]
        self.test_sents = None  # list[list[str]]
        self.test_tags = None  # list[list[str]]

    def load_data(self, filename, train):
        """Read the CoNLL-U data in the file.

        Multi-word tokens, and any token whose form == None, are skipped.

        If train==True, set the following class variables:

        train_sents : list[list[str]]
            training sentences as a list of lists of word forms,
            with EOS marker added to each sentence
        train_tags: list[list[str]]
            corresponding POS tags for train_sents,
            with artificial END tag added to each inner list
        vocab : list[str]
            list of *sorted* vocabulary words in the *training* data,
            including EOS and UNK (for oov words in test data)
        pos_tagset : list[str]
            list of *sorted* POS tags in the *training* data,
            including the artificial END tag

        If train==False (it's test data), set the following class variables:

        test_sents : list[list[str]]
            testing sentences as a list of lists of word forms,
            with EOS marker added to each sentence
        test_tags: list[list[str]]
            corresponding POS tags for test_sents,
            with artificial END tag added to each inner list

        Parameters
        ----------
        filename : str
            path to the CoNLL-U file
        train : bool
            if True, set the training sentences with corresponding tags, vocab, and tagset;
            otherwise set the testing sentences with corresponding tags
        """
        # initialize instance variables
        if train:
            self.train_sents = []  
            self.train_tags = []   
            self.vocab = []         
            self.pos_tagset = []   
            vocab = set()
            pos_tag_set = set()
        self.test_sents = []   
        self.test_tags = []    

        # read the conllu file
        sentences = pyconll.iter_from_file(filename)

        for sentence in sentences:
            forms = []
            tags = []
            for token in sentence:
                # skip multiword and tokens with form None
                if not (token.is_multiword() or token.form == None): 
                    forms.append(token.form)
                    tags.append(token.upos)
                    if train:
                        vocab.add(token.form)
                        pos_tag_set.add(token.upos)
            # add EOS marker and END tag    
            forms.append(EOS)           
            tags.append(END)  

            # set training sentences and tags
            if train:
                self.train_sents.append(forms)
                self.train_tags.append(tags)

            # set testing sentences and tags
            else:
                #add the processed sentence
                self.test_sents.append(forms)
                self.test_tags.append(tags)

        
        if train:
            #  add END tag and sort tagset   
            self.pos_tagset = list(pos_tag_set)
            self.pos_tagset.append(END)
            self.pos_tagset.sort()

            # add UNK, EOS tokens and sort vocabulary
            self.vocab = list(vocab)
            self.vocab.append(UNK)
            self.vocab.append(EOS)
            self.vocab.sort()                      

       
    def get_mapping(self, str_list, reverse=False):
        """Return a mapping, as a dictionary, from the indices of str_list to
        their values ({idx:str}). If reverse=True, the mapping has the values in str_list
        as keys and the index as value ({str:idx})

        Parameters
        ----------
        str_list : list[str]
            the list to map
        reverse : bool, optional
            if True, map strings to indices {str:idx},
            otherwise {idx:str} (default is False)

        Returns
        -------
        dict
            A dictionary mapping {idx:str}, or {str:idx} if reverse=True
        """
        dict = {}
        for idx, str in enumerate(str_list):
            if reverse == True:
                dict[str] = idx
            else:
                dict[idx] = str            
        return dict

    def replace_oov(self):
        """Return a list[list[str]] in which the OOV words (words not in vocab)
        in test sentences are replaced with UNK.

        Returns
        -------
        list[list[str]]
            test sentences with oov words replaced by UNK
        """
        for sent in self.test_sents:
            for idx, word in enumerate(sent):
                if word not in self.vocab:
                    sent[idx] = UNK
        return self.test_sents

    def evaluate_model(self, test_file):
        """Evaluate the trained HMM on the CoNNL-U sentences in test_file.

        Use the load_data() function to load the
        testing data. After loading the test data, access the
        testing sentences and their corresponding tags with
        self.test_sents and self.test_tags.

        Returns a full report of the evaluation (using scikit-learn),
        the macro f1-score.

        Parameters
        ----------
        test_file: str
            CoNLL-U testing file

        Returns
        -------
        (str, float)
            report : str
                full evaluation report as returned by classification_report()
            f1_score : float
                macro f1-score of the test data
        """
        # load the test data, replace oov words
        self.load_data(test_file, train=False)
        self.replace_oov()

        # create a nested list of predicted POS tags for each test sentence
        pos_pred = [self.decode(sent) for sent in self.test_sents]

        # flatten the arrays
        flat_pred = [word for sent in pos_pred for word in sent]
        flat_true = [word for sent in self.test_tags for word in sent]

        # create and return report and f1-score
        report_dict = classification_report(flat_true, flat_pred, output_dict=True)
        report_str = classification_report(flat_true, flat_pred, output_dict=False)
        return report_str, report_dict['macro avg']['f1-score']


    def train_model(self, train_file):
        """ Do not implement this method here. It should be implemented
        in the child classes.
        """
        raise NotImplementedError("should be implemented in child class.")

    def decode(self, sentence):
        """ Do not implement this method here. It should be implemented
        in the child classes.
        """
        raise NotImplementedError("should be implemented in child class.")
