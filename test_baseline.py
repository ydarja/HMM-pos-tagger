import unittest
from tagger import UNK, EOS, END
from baseline import *


class TestBaseline(unittest.TestCase):

    def setUp(self) -> None:
        self.bl = Baseline()

    def tearDown(self) -> None:
        self.bl = None

    def test_train1(self):
        expected = {
            '.': 'PUNCT',
            '</s>': 'END',
            '<UNK>': 'AUX',
            'But': 'CCONJ',
            'He': 'PRON',
            'We': 'PRON',
            'believe': 'VERB',
            'ca': 'AUX',
            'denied': 'VERB',
            'do': 'AUX',
            'has': 'AUX',
            'have': 'VERB',
            'him': 'PRON',
            'it': 'PRON',
            "n't": 'PART',
            'prove': 'VERB',
            'this': 'PRON',
            'to': 'PART',
            'we': 'PRON'
        }
        self.bl.train_model('unittest_train.conllu')
        actual = self.bl.word_pos_map
        self.assertDictEqual(expected, actual)
        
        print("EXPECTED")
        for k,v in expected.items():
            print(k,":",v)
        print("ACTUAL")
        for k,v in actual.items():
            print(k,":",v)
       
    def test_train2(self):
        expected = {'</s>': 'END',
                    '<UNK>': 'AUX',
                    'can': 'AUX',
                    'mary': 'PROPN',
                    'see': 'VERB',
                    'spot': 'PROPN',
                    'will': 'AUX'}
        self.bl.train_model('unittest_train2.conllu')
        actual = self.bl.word_pos_map
        self.assertDictEqual(expected, actual)

    def test_decode1(self):
        self.bl.word_pos_map = {
            '.': 'PUNCT',
            '</s>': 'END',
            '<UNK>': 'AUX',
            'But': 'CCONJ',
            'He': 'PRON',
            'We': 'PRON',
            'believe': 'VERB',
            'ca': 'AUX',
            'denied': 'VERB',
            'do': 'AUX',
            'has': 'AUX',
            'have': 'VERB',
            'him': 'PRON',
            'it': 'PRON',
            "n't": 'PART',
            'prove': 'VERB',
            'this': 'PRON',
            'to': 'PART',
            'we': 'PRON'
        }
        sent = ["He", "denied", "it", "."]
        expected = ['PRON', 'VERB', 'PRON', 'PUNCT']
        actual = self.bl.decode(sent)
        self.assertListEqual(expected, actual)

if __name__ == '__main__':
    unittest.main()
