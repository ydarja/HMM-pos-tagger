import unittest
from tagger import UNK, EOS, END
from hmm import *


class TestHMM(unittest.TestCase):

    def setUp(self) -> None:
        self.hmm = HMM()
        self.pi = np.array([0.1, 0.2, 0.1, 0.1, 0.3, 0.1, 0.1])
        self.trans_mtx = np.array([
            [0.1, 0.1, 0.1, 0.3, 0.1, 0.1, 0.2],
            [0.125, 0.125, 0.125, 0.125, 0.25, 0.125, 0.125],
            [0.14285714, 0.14285714, 0.14285714, 0.14285714, 0.14285714, 0.14285714, 0.14285714],
            [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.4],
            [0.30769231, 0.07692308, 0.07692308, 0.07692308, 0.07692308, 0.30769231, 0.07692308],
            [0.1, 0.1, 0.4, 0.1, 0.1, 0.1, 0.1],
            [0.09090909, 0.09090909, 0.09090909, 0.18181818, 0.36363636, 0.09090909, 0.09090909]
        ])
        self.em_mtx = np.array([
            [0.04545455, 0.04545455, 0.04545455, 0.04545455, 0.04545455, 0.04545455,
            0.04545455, 0.09090909, 0.04545455, 0.09090909, 0.09090909, 0.04545455,
            0.04545455, 0.04545455, 0.04545455, 0.04545455, 0.04545455, 0.04545455,
            0.04545455],
            [0.05,       0.05,       0.05,       0.1,        0.05,       0.05,
            0.05,       0.05,       0.05,       0.05,       0.05,       0.05,
            0.05,       0.05,       0.05,       0.05,       0.05,       0.05,
            0.05],
            [0.04545455, 0.18181818, 0.04545455, 0.04545455, 0.04545455, 0.04545455, 0.04545455,
            0.04545455, 0.04545455, 0.04545455, 0.04545455, 0.04545455, 0.04545455, 0.04545455,
            0.04545455, 0.04545455, 0.04545455, 0.04545455, 0.04545455],
            [0.04545455, 0.04545455, 0.04545455, 0.04545455, 0.04545455, 0.04545455,
            0.04545455, 0.04545455, 0.04545455, 0.04545455, 0.04545455, 0.04545455,
            0.04545455, 0.04545455, 0.13636364, 0.04545455, 0.04545455, 0.09090909,
            0.04545455],
            [0.04, 0.04, 0.04, 0.04, 0.08, 0.08, 0.04, 0.04, 0.04, 0.04,
            0.04, 0.04, 0.08, 0.08, 0.04, 0.04, 0.08, 0.04, 0.08],
            [0.18181818, 0.04545455, 0.04545455, 0.04545455, 0.04545455, 0.04545455,
            0.04545455, 0.04545455, 0.04545455, 0.04545455, 0.04545455, 0.04545455,
            0.04545455, 0.04545455, 0.04545455, 0.04545455, 0.04545455, 0.04545455,
            0.04545455],
            [0.04347826, 0.04347826, 0.04347826, 0.04347826, 0.04347826, 0.04347826,
            0.08695652, 0.04347826, 0.08695652, 0.04347826, 0.04347826, 0.08695652,
            0.04347826, 0.04347826, 0.04347826, 0.08695652, 0.04347826, 0.04347826,
            0.04347826]
        ])
        self.idx_tag_map = {0: 'AUX', 1: 'CCONJ', 2: 'END', 3: 'PART', 4: 'PRON', 5: 'PUNCT', 6: 'VERB'}
        self.word_idx_map = {'.': 0, '</s': 1, '<UNK>': 2, 'But': 3, 'He': 4, 'We': 5,
                        'believe': 6, 'ca': 7, 'denied': 8, 'do': 9, 'has': 10, 'have': 11,
                        'him': 12, 'it': 13, "n't": 14, 'prove': 15, 'this': 16, "to": 17, 'we': 18}

    def tearDown(self) -> None:
        self.hmm = None

    def test_initial_probs1(self):
        self.hmm.train_tags = [
            ['N', 'M', 'V', END],
            ['N', 'M', 'V', 'N', END]
        ]
        self.hmm.tag_idx_map = {'N': 0,
                                'V': 1,
                                'M': 2,
                                END: 3}
        self.hmm.k = 1.0
        expected = np.array([3/6, 1/6, 1/6, 1/6])
        self.hmm.initial_probs()
        actual = self.hmm.pi
        self.assertTrue(np.allclose(expected, actual, rtol=1e-05, atol=1e-08),
                        msg=f"\nexpected:\n{expected}\nactual:\n{actual}")

    def test_initial_probs2(self):
        self.hmm.train_tags = [
            ['N', 'N', 'M', 'V', 'N', END],
            ['N', 'M', 'V', 'N', END],
            ['M', 'N', 'V', 'N', END],
            ['N', 'M', 'V', 'N', END]
        ]
        self.hmm.tag_idx_map = {'N': 0,
                                'M': 1,
                                'V': 2,
                                END: 3}
        self.hmm.k = 1.0
        expected = np.array([0.5, 0.25, 0.125, 0.125])
        self.hmm.initial_probs()
        actual = self.hmm.pi
        self.assertTrue(np.allclose(expected, actual, rtol=1e-05, atol=1e-08),
                        msg=f"\nexpected:\n{expected}\nactual:\n{actual}")

    def test_transition_matrix1(self):
        self.hmm.train_tags = [
            ['N', 'M', 'V', END],
            ['N', 'M', 'V', 'N', END]
        ]
        self.hmm.tag_idx_map = {END: 0,
                                'N': 1,
                                'V': 2,
                                'M': 3}
        self.hmm.k = 1.0
        expected = np.array([
            [0.25,       0.25,       0.25,       0.25],
            [0.28571429, 0.14285714, 0.14285714, 0.42857143],
            [0.33333333, 0.33333333, 0.16666667, 0.16666667],
            [0.16666667, 0.16666667, 0.5,        0.16666667]
        ])
        self.hmm.transition_matrix()
        actual = self.hmm.transition_mtx
        self.assertTrue(np.allclose(expected, actual, rtol=1e-03, atol=1e-03),
                        msg=f"\nexpected:\n{expected}\nactual:\n{actual}")

    def test_transition_matrix2(self):
        self.hmm.train_tags = [
            ['N', 'N', 'M', 'V', 'N', END],
            ['N', 'M', 'V', 'N', END],
            ['M', 'N', 'V', 'N', END],
            ['N', 'M', 'V', 'N', END]
        ]
        self.hmm.tag_idx_map = {'N': 0, 'M': 1, 'V': 2, END: 3}
        self.hmm.k = 1.0
        expected = np.array([
            [0.15384615, 0.30769231, 0.15384615, 0.38461538],
            [0.25,       0.125,      0.5,        0.125],
            [0.625,      0.125,      0.125,      0.125],
            [0.25,       0.25,       0.25,       0.25]
        ])
        self.hmm.transition_matrix()
        actual = self.hmm.transition_mtx
        self.assertTrue(np.allclose(expected, actual, rtol=1e-03, atol=1e-03),
                        msg=f"\nexpected:\n{expected}\nactual:\n{actual}")

    def test_emission_matrix1(self):
        self.hmm.train_sents = [
            ['mary', 'can', 'spot', EOS],
            ['spot', 'will', 'spot', 'will', EOS]
        ]
        self.hmm.train_tags = [
            ['N', 'M', 'V', END],
            ['N', 'M', 'V', 'N', END]
        ]
        self.hmm.word_idx_map = {'can': 0,
                        'mary': 1,
                        'spot': 2,
                        'will': 3,
                        EOS: 4
                        }
        self.hmm.tag_idx_map = {END: 0,
                       'N': 1,
                       'V': 2,
                       'M': 3}
        self.hmm.k = 1.0
        expected = np.array([
            [0.14285714, 0.14285714, 0.14285714, 0.14285714, 0.42857143],
            [0.125,      0.25,       0.25,       0.25,       0.125],
            [0.14285714, 0.14285714, 0.42857143, 0.14285714, 0.14285714],
            [0.28571429, 0.14285714, 0.14285714, 0.28571429, 0.14285714]
        ])
        self.hmm.emission_matrix()
        actual = self.hmm.emission_mtx
        self.assertTrue(np.allclose(expected, actual, rtol=1e-03, atol=1e-03),
                        msg=f"\nexpected:\n{expected}\nactual:\n{actual}")

    def test_emission_matrix2(self):
        self.hmm.train_sents = [
            ['mary', 'jane', 'can', 'see', 'will', EOS],
            ['spot', 'will', 'see', 'mary', EOS],
            ['will', 'pat', 'spot', 'mary', EOS],
            ['mary', 'will', 'pat', 'spot', EOS]
        ]
        self.hmm.train_tags = [
            ['N', 'N', 'M', 'V', 'N', END],
            ['N', 'M', 'V', 'N', END],
            ['M', 'N', 'V', 'N', END],
            ['N', 'M', 'V', 'N', END]
        ]
        self.hmm.word_idx_map = {'mary': 0, 'jane': 1, 'will': 2, 'spot': 3,
                        'can': 4, 'see': 5, 'pat': 6, EOS: 7}
        self.hmm.tag_idx_map = {'N': 0, 'M': 1, 'V': 2, END: 3}
        self.hmm.k = 1.0
        expected = np.array([
            [0.29411765, 0.11764706, 0.11764706, 0.17647059, 0.05882353, 0.05882353, 0.11764706, 0.05882353],
            [0.08333333, 0.08333333, 0.33333333, 0.08333333, 0.16666667, 0.08333333, 0.08333333, 0.08333333],
            [0.08333333, 0.08333333, 0.08333333, 0.16666667, 0.08333333, 0.25,       0.16666667, 0.08333333],
            [0.08333333, 0.08333333, 0.08333333, 0.08333333, 0.08333333, 0.08333333, 0.08333333, 0.41666667]
        ])
        self.hmm.emission_matrix()
        actual = self.hmm.emission_mtx
        self.assertTrue(np.allclose(expected, actual, rtol=1e-03, atol=1e-03),
                        msg=f"\nexpected:\n{expected}\nactual:\n{actual}")

    def test_train1(self):
        expected = self.pi
        self.hmm.train_model('unittest_train.conllu')
        actual = self.hmm.pi
        self.assertTrue(np.allclose(expected, actual, rtol=1e-05, atol=1e-08),
                        msg=f"\nexpected pi:\n{expected}\nactual pi:\n{actual}")

    def test_train2(self):
        expected = self.trans_mtx
        self.hmm.train_model('unittest_train.conllu')
        actual = self.hmm.transition_mtx
        self.assertTrue(np.allclose(expected, actual, rtol=1e-05, atol=1e-08),
                        msg=f"\nexpected transition mtx:\n{expected}\nactual transition mtx:\n{actual}")

    def test_train3(self):
        expected = self.em_mtx
        self.hmm.train_model('unittest_train.conllu')
        actual = self.hmm.emission_mtx
        self.assertTrue(np.allclose(expected, actual, rtol=1e-05, atol=1e-08),
                        msg=f"\nexpected emission mtx:\n{expected}\nactual emission mtx:\n{actual}")

    def test_decode1(self):
        # sentences = [
        #     ['mary', 'can', 'spot', EOS],
        #     ['spot', 'will', 'spot', 'will', EOS]
        # ]
        # tagged_sentences = [
        #     ['N', 'M', 'V', END],
        #     ['N', 'M', 'V', 'N', END]
        # ]
        self.hmm.word_idx_map = {'mary': 0,
                        'can': 1,
                        'spot': 2,
                        'will': 3,
                        EOS: 4
                        }
        self.hmm.idx_tag_map = {0: 'N',
                       1: 'V',
                       2: 'M',
                       3: END}
        self.hmm.pi = np.array([3/6, 1/6, 1/6, 1/6])
        self.hmm.transition_mtx = np.array([
            [1/7, 1/7, 3/7, 2/7],
            [2/6, 1/6, 1/6, 2/6],
            [1/6, 3/6, 1/6, 1/6],
            [1/4, 1/4, 1/4, 1/4]
        ])
        self.hmm.emission_mtx = np.array([
            [2/8, 1/8, 2/8, 2/8, 1/8],
            [1/7, 1/7, 3/7, 1/7, 1/7],
            [1/7, 2/7, 1/7, 2/7, 1/7],
            [1/7, 1/7, 1/7, 1/7, 3/7]
        ])

        sent = ['spot', 'can', 'spot', 'mary']
        expected = ['N', 'M', 'V', 'N']
        actual = self.hmm.decode(sent)
        self.assertListEqual(expected, actual)


if __name__ == '__main__':
    unittest.main()
