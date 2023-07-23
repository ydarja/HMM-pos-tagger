"""
 Course:        Statistical Language Processing - Summer 2023
 Assignment:    (A3)
 Author:        (Daniel Stuhlinger, Darja Jepifanpva)

 Honor Code:    We pledge that this program represents our own work.
"""
from tagger import Tagger
import numpy as np


class HMM(Tagger):
    def __init__(self):
        super().__init__()

        self.k = None  # float, for smoothing
        self.tag_idx_map = None  # {tag: idx} dictionary
        self.word_idx_map = None  # {word: idx} dictionary
        self.idx_tag_map = None  # {idx: tag} dictionary
        self.pi = None  # array of length num_tags
        self.transition_mtx = None  # matrix of shape (num_tags, num_tags)
        self.emission_mtx = None  # matrix of shape (num_tags, len(vocab))

    def initial_probs(self):
        """Calculate the probability distribution of POS tags
        of the first words in the training data (self.train_tags). Referred to
        as 'pi' in J&M.

        Set the class variable pi:
            - a numpy array of shape (ns,), where ns is the number of states,
              in this case the number of POS tags.
            - Each value in pi represents the probability of a POS tag occurring
              as the first tag of a training sentence (self.train_tags)
            - Use self.tag_idx_map (created in train_model()) to determine the
              index for each POS probability.
            - Apply Add-k smoothing (k is a class variable set by train_model())

        """
        # create a frequency dictionary with each first word in sentences as keys
        tag_freq = dict()
        for sent in self.train_tags:
            if sent[0] not in tag_freq:
                tag_freq[sent[0]] = 1
            else:
                tag_freq[sent[0]] += 1

        # add k smoothing
        for pos in self.tag_idx_map:
            if pos not in tag_freq:
                tag_freq[pos] = self.k
            else:
                tag_freq[pos] += self.k

        # create and fill in a probability dictionary
        tag_prob = dict()
        for pos, freq in tag_freq.items():
            tag_prob[pos] = freq / (len(self.train_tags) + (len(self.tag_idx_map) * self.k))

        # pi dict with tags' indices as keys and probability of that tag strarting a sentence as values 
        self.pi = np.full(len(tag_freq), fill_value=0.)
        for tag, prob in tag_prob.items():
            self.pi[self.tag_idx_map[tag]] += prob


    def transition_matrix(self):
        """Calculates the transition probability matrix, also referred to
        as simply 'a' or 'A' in J&M ch. 8.

        Set the class variable transition_mtx:
            - a numpy array of shape (ns,ns), where ns is the number of states
              in the HMM (in this case, the number of unique POS tags in train_tags).
            - Each cell in the matrix represents the probability of transitioning
              from the POS at row to the POS at col. Row and column indexing is
              according to the tag_idx_map.
            - Apply Add-k smoothing (k is class variable set by train_model())

        """
        # create and fill the matrix of POS bigrams counts
        bigram_counts = np.full((len(self.tag_idx_map), len(self.tag_idx_map)), self.k)
        for sent in self.train_tags:
            for i in range(1, len(sent)):
                idx_row = self.tag_idx_map[sent[i-1]]
                idx_col = self.tag_idx_map[sent[i]]
                bigram_counts[idx_row][idx_col] +=1

        # create and fill the matrix of POS bigram probabilities
        bigram_probs = np.zeros(bigram_counts.shape, dtype=float)
        for i in range(len(self.tag_idx_map)):
            for j in range(len(self.tag_idx_map)):
                bigram_probs[i][j] = bigram_counts[i][j] / np.sum(bigram_counts[i,:])

        # set the instance variable
        self.transition_mtx = bigram_probs   

    def emission_matrix(self):
        """Calculates the emission probability matrix, also referred to
        as simply 'b' or 'B' in J&M ch. 8.

        Set the class variable emission_mtx:
            - a numpy array of shape (n_s, n_o), where n_s is the number of states
              in the HMM (the number of unique POS tags in the training data),
              and n_o is the number of unique observations (words in the vocabulary).
            - Each cell in the matrix represents the likelihood of word in col,
              given POS in row: P(w|POS).
            - Row indexing is according to self.tag_idx_map
            - Column indexing is according to self.word_idx_map
            - Apply Add-k smoothing (k is class variable set by train_model())

        """
        # initialize counts matrix with add-k smoothing
        counts = np.full((len(self.tag_idx_map), len(self.word_idx_map)), self.k)

        # flatten the arrays of words and tags
        flattened_sents = [word for sent in self.train_sents for word in sent]
        flattened_tags = [tag for sent in self.train_tags for tag in sent]
        # create a list of tuples (word, pos)
        word_pos = list(zip(flattened_sents, flattened_tags))

        # fill in counts matrix of word having a certain POS tag
        for word, pos in word_pos:
            counts[self.tag_idx_map[pos]][self.word_idx_map[word]] += 1

        # create and fill probability matrix of word given a POS tag    
        probs = np.full((len(self.tag_idx_map), len(self.word_idx_map)), 0.)
        for i in range(len(self.tag_idx_map)):
            for j in range(len(self.word_idx_map)):
                probs[i][j] = counts[i][j] / np.sum(counts[i,:])

        # set the instance variable
        self.emission_mtx = probs

    def train_model(self, train_file, k=1.0):
        """Train the HMM using the CoNLL-U data in the training file,
        applying Add-k smoothing.

        Use the load_data() function in the parent class (Tagger) to load the
        training data. The class variables set by Tagger.load_data() are
        available here as well. For example, access the vocabulary with self.vocab.

        Using the functions implemented above, set the class variables
        for all components of the HMM:
            - k
            - idx_tag_map (used by the decoder)
            - tag_idx_map
            - word_idx_map
            - pi
            - transition_mtx
            - emission_mtx

        Parameters
        ----------
        train_file: str
            CoNLL-U training file
        k: float
            optional smoothing value, default 1.0
        """
        # load the data 
        self.load_data(train_file, train=True)

        # set k
        self.k = k

        # create idx_tag and tag_idx dictionary
        self.idx_tag_map = {}
        self.tag_idx_map = {}
        for idx, tag in enumerate(self.pos_tagset):
            self.idx_tag_map[idx] = tag   
            self.tag_idx_map[tag] = idx         
        
        # create word_idx dictionary
        self.word_idx_map = {}
        for idx, word in enumerate(self.vocab):
            self.word_idx_map [word] = idx 
  
        # set pi
        self.initial_probs()

        # set transition and emission matrices
        self.transition_matrix()
        self.emission_matrix()

    def decode(self, sentence):
        """Use the Viterbi algorithm to find the best POS tagging
        for the input sentence.
        Parameters
        ----------
        sentence : list[str]
            The sentence to tag

        Returns
        -------
        pred_tags : list[str]
            predicted POS tags, one tag per word in input sentence
        """
        # In the instructions PDF it is said that this method should return the pos tags as well as their probability.
        # But in the Pseudo Code the line is commented out and I think the unittests would also have to be adjusted
        # if the probability is returned. So we decided to not return it here.

        # initialize path-probability-matrix
        path_prob = np.zeros((len(self.idx_tag_map), len(sentence)), dtype=float)
        # initialize back-pointer-matrix
        back_pointer_mtx = np.zeros((len(self.idx_tag_map), len(sentence)), dtype=int)

        # fill in the initial probabilities into the first column of path_prob
        for idx, _ in self.idx_tag_map.items():
            path_prob[idx, 0] = self.pi[idx] * self.emission_mtx[idx][self.word_idx_map[sentence[0]]]

        # iterate over words
        for t in range(1, len(sentence)):
            # iterate over pos tags
            for s in range(len(self.idx_tag_map)):
                max_prob = 0
                # determine which previous pos tag yields the maximum probability for current cell
                for s_prime in range(len(self.idx_tag_map)):
                    # calculate the three needed probability and their joint probability
                    previous_prob = path_prob[s_prime][t-1]
                    transition_prob = self.transition_mtx[s_prime][s]
                    emission_prob = self.emission_mtx[s][self.word_idx_map[sentence[t]]]
                    joint_prob = previous_prob * transition_prob * emission_prob
                    # if the current probability is higher than max, replace it
                    # and insert row index that yields maximum into the back-pointer-matrix
                    if joint_prob > max_prob:
                        max_prob = joint_prob
                        back_pointer_mtx[s, t] = s_prime
                # keep maximum probability in the current cell
                path_prob[s][t] = max_prob
        pos_list = []
        # determine max probability in last column
        max_prob_idx = np.argmax(path_prob[:, -1])
        pos_list.append(self.idx_tag_map[max_prob_idx])
        _, num_col = path_prob.shape
        # recover the path that yields the maximum probability from back pointer matrix
        for i in range(1, num_col):
            pos_list.append(self.idx_tag_map[back_pointer_mtx[max_prob_idx][-i]])
            max_prob_idx = back_pointer_mtx[max_prob_idx][-i]
        pos_list.reverse()
        return pos_list


def main(train_file, test_file, k=1.0):
    """Train and test the HMM tagger with the input files.
    Print the evaluation report and the macro f1-score.

    Parameters
    ----------
    train_file : str
        conllu file to use for training
    test_file : str
        conllu file to use for testing
    k : float
        value for Add-k smoothing, default 1.0
    """
    m1 = HMM()
    # train the model
    m1.train_model(train_file, k=k)
    # evaluate the model
    report_str, f1_score = m1.evaluate_model(test_file)
    print(f'Report:\n{report_str}\nF1 Score = {f1_score}')




if __name__ == '__main__':
    # best performance we could get is with k=0.04
    main('Data/en_ewt-ud-train.conllu', 'Data/en_ewt-ud-test.conllu', k=1.0)

