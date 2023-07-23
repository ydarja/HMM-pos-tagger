import unittest
from tagger import *


class TestTagger(unittest.TestCase):

    def setUp(self) -> None:
        self.tagger = Tagger()

    def tearDown(self) -> None:
        self.tagger = None

    def test_load_data1(self):
        expected = [['He', 'has', 'denied', 'this', '.', EOS],
                    ['We', 'do', "n't", 'have', 'to', 'believe', 'him', '.', EOS],
                    ['But', 'we', 'ca', "n't", 'prove', 'it', '.', EOS]]
        self.tagger.load_data('unittest_train.conllu', train=True)
        actual = self.tagger.train_sents



    def test_load_data2(self):
        expected = [['PRON', 'AUX', 'VERB', 'PRON', 'PUNCT', END],
                    ['PRON', 'AUX', 'PART', 'VERB', 'PART', 'VERB', 'PRON', 'PUNCT', END],
                    ['CCONJ', 'PRON', 'AUX', 'PART', 'VERB', 'PRON', 'PUNCT', END]]
        self.tagger.load_data('unittest_train.conllu', train=True)
        actual = self.tagger.train_tags
        self.assertListEqual(expected, actual)

    def test_load_data3(self):
        expected = ['.', EOS, '<UNK>', 'But', 'He', 'We', 'believe', 'ca', 'denied', 'do',
                    'has', 'have', 'him', 'it', "n't", 'prove', 'this', 'to', 'we']
        self.tagger.load_data('unittest_train.conllu', train=True)
        actual = self.tagger.vocab
        self.assertListEqual(expected, actual)
        
    def test_load_data4(self):
        expected = ['AUX', 'CCONJ', END, 'PART', 'PRON', 'PUNCT', 'VERB']
        self.tagger.load_data('unittest_train.conllu', train=True)
        actual = self.tagger.pos_tagset
        self.assertListEqual(expected, actual)
       

    def test_get_mapping1(self):
        lst = ['.', 'He', 'denied', 'has', 'this', EOS]
        expected = {0: '.',
                    1: 'He',
                    2: 'denied',
                    3: 'has',
                    4: 'this',
                    5: EOS
                    }
        actual = self.tagger.get_mapping(lst, reverse=False)
        self.assertDictEqual(expected, actual)

    def test_get_mapping2(self):
        lst = ['.', 'He', 'denied', 'has', 'this', EOS]
        expected = {'.': 0 ,
                    'He': 1,
                    'denied': 2,
                    'has': 3,
                    'this': 4,
                    EOS: 5
                    }
        actual = self.tagger.get_mapping(lst, reverse=True)
        self.assertDictEqual(expected, actual)

    def test_replace_oov(self):
        self.tagger.test_sents = [['pat', 'will', 'walk', 'spot', EOS],
                      ['joe', 'can', 'train', 'spot', EOS]]
        self.tagger.vocab = ['pat', 'will', 'spot', 'can', EOS]
        expected = [['pat', 'will', UNK, 'spot', EOS],
                    [UNK, 'can', UNK, 'spot', EOS]]
        actual = self.tagger.replace_oov()
        self.assertListEqual(expected, actual)

if __name__ == '__main__':
    unittest.main()

