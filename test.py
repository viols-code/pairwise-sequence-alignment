import io
import unittest
import numpy as np
from unittest.mock import patch

from sequence_alignment import compute_smith_waterman, determine_step, traceback_smith_waterman, \
    compute_needleman_wunsch, path


class PairwiseAlignments(unittest.TestCase):

    def test_determine_step_1(self):
        prev = np.argmax((0, 1, 1, 1))
        self.assertEqual(7, determine_step(prev, 1, 1, 1))

    def test_determine_step_2(self):
        prev = np.argmax((0, -1, -1, -1))
        self.assertEqual(0, determine_step(prev, -1, -1, -1))

    def test_determine_step_3(self):
        prev = np.argmax((0, 0, -1, 0))
        self.assertEqual(0, determine_step(prev, 0, -1, 0))

    def test_determine_step_4(self):
        prev = np.argmax((0, 0, 0, 0))
        self.assertEqual(0, determine_step(prev, 0, 0, 0))

    def test_determine_step_5(self):
        prev = np.argmax((0, 1, 0, 0))
        self.assertEqual(1, determine_step(prev, 1, 0, 0))

    def test_determine_step_6(self):
        prev = np.argmax((0, 1, 0, 1))
        self.assertEqual(5, determine_step(prev, 1, 0, 1))

    def test_determine_step_7(self):
        prev = np.argmax((0, 1, 1, 1))
        self.assertEqual(7, determine_step(prev, 1, 1, 1))

    def test_determine_step_8(self):
        prev = np.argmax((0, 0, 1, 1))
        self.assertEqual(6, determine_step(prev, 0, 1, 1))

    def test_determine_step_9(self):
        prev = np.argmax((0, 0, 0, 1))
        self.assertEqual(3, determine_step(prev, 0, 0, 1))

    def test_determine_step_10(self):
        prev = np.argmax((0, 0, 1, 0))
        self.assertEqual(2, determine_step(prev, 0, 1, 0))

    def test_determine_step_11(self):
        prev = np.argmax((0, 1, 1, 0))
        self.assertEqual(4, determine_step(prev, 1, 1, 0))

    def test_compute_smith_waterman_1(self):
        seq1 = "AATCG"
        seq2 = "AACG"
        matrix, traceback = compute_smith_waterman(1, -1, -2, seq1, seq2, 1)
        np.testing.assert_array_equal(matrix, np.array([[0, 0, 0, 0, 0],
                                                        [0, 1, 1, 0, 0],
                                                        [0, 1, 2, 0, 0],
                                                        [0, 0, 0, 1, 0],
                                                        [0, 0, 0, 1, 0],
                                                        [0, 0, 0, 0, 2]]))
        np.testing.assert_array_equal(traceback, np.array([[0, 0, 0, 0, 0],
                                                           [0, 1, 1, 0, 0],
                                                           [0, 1, 1, 0, 0],
                                                           [0, 0, 0, 1, 0],
                                                           [0, 0, 0, 1, 0],
                                                           [0, 0, 0, 0, 1]]))

    def test_compute_smith_waterman_2(self):
        seq1 = "CIAO"
        seq2 = "CIAOCI"
        matrix, traceback = compute_smith_waterman(2, -1, -2, seq1, seq2, 1)
        np.testing.assert_array_equal(matrix, np.array([[0, 0, 0, 0, 0, 0, 0],
                                                        [0, 2, 0, 0, 0, 2, 0],
                                                        [0, 0, 4, 2, 0, 0, 4],
                                                        [0, 0, 2, 6, 4, 2, 2],
                                                        [0, 0, 0, 4, 8, 6, 4]]))
        np.testing.assert_array_equal(traceback, np.array([[0, 0, 0, 0, 0, 0, 0],
                                                           [0, 1, 0, 0, 0, 1, 0],
                                                           [0, 0, 1, 3, 0, 0, 1],
                                                           [0, 0, 2, 1, 3, 3, 2],
                                                           [0, 0, 0, 2, 1, 3, 3]]))

    def test_compute_smith_waterman_3(self):
        seq1 = "ACACACC"
        seq2 = "ACA"
        matrix, traceback = compute_smith_waterman(3, -1, -1, seq1, seq2, 1)
        np.testing.assert_array_equal(matrix, np.array([[0, 0, 0, 0],
                                                        [0, 3, 2, 3],
                                                        [0, 2, 6, 5],
                                                        [0, 3, 5, 9],
                                                        [0, 2, 6, 8],
                                                        [0, 3, 5, 9],
                                                        [0, 2, 6, 8],
                                                        [0, 1, 5, 7]]))
        np.testing.assert_array_equal(traceback, np.array([[0, 0, 0, 0],
                                                           [0, 1, 3, 1],
                                                           [0, 2, 1, 3],
                                                           [0, 1, 2, 1],
                                                           [0, 2, 1, 2],
                                                           [0, 1, 2, 1],
                                                           [0, 2, 1, 2],
                                                           [0, 2, 4, 2]]))

    def test_compute_smith_waterman_4(self):
        seq1 = "ACACACC"
        seq2 = "ACA"
        matrix, traceback = compute_smith_waterman(3, -1, -1, seq1, seq2, 0)
        np.testing.assert_array_equal(matrix, np.array([[0, 0, 0, 0],
                                                        [0, 3, 2, 3],
                                                        [0, 2, 6, 5],
                                                        [0, 3, 5, 9],
                                                        [0, 2, 6, 8],
                                                        [0, 3, 5, 9],
                                                        [0, 2, 6, 8],
                                                        [0, 1, 5, 7]]))
        np.testing.assert_array_equal(traceback, np.array([[0, 0, 0, 0],
                                                           [0, 1, 3, 1],
                                                           [0, 2, 1, 3],
                                                           [0, 1, 2, 1],
                                                           [0, 2, 1, 2],
                                                           [0, 1, 2, 1],
                                                           [0, 2, 1, 2],
                                                           [0, 2, 1, 2]]))

    def test_compute_smith_waterman_5(self):
        seq1 = "TGCT"
        seq2 = "ATTCA"
        matrix, traceback = compute_smith_waterman(3, -1, -3, seq1, seq2, 1)
        np.testing.assert_array_equal(matrix, np.array([[0, 0, 0, 0, 0, 0],
                                                        [0, 0, 3, 3, 0, 0],
                                                        [0, 0, 0, 2, 2, 0],
                                                        [0, 0, 0, 0, 5, 2],
                                                        [0, 0, 3, 3, 2, 4]]))
        np.testing.assert_array_equal(traceback, np.array([[0, 0, 0, 0, 0, 0],
                                                           [0, 0, 1, 1, 0, 0],
                                                           [0, 0, 0, 1, 1, 0],
                                                           [0, 0, 0, 0, 1, 3],
                                                           [0, 0, 1, 1, 2, 1]]))
        matrix2, traceback2 = compute_smith_waterman(3, -1, -3, seq1, seq2, 0)
        np.testing.assert_array_equal(matrix, matrix2)
        np.testing.assert_array_equal(traceback, traceback2)

    def test_compute_needleman_wunsch_1(self):
        seq1 = "AATCG"
        seq2 = "AACG"
        matrix, traceback = compute_needleman_wunsch(1, -1, -2, seq1, seq2, 1)
        np.testing.assert_array_equal(matrix, np.array([[0, -2, -4, -6, -8],
                                                        [-2, 1, -1, -3, -5],
                                                        [-4, -1, 2, 0, -2],
                                                        [-6, -3, 0, 1, -1],
                                                        [-8, -5, -2, 1, 0],
                                                        [-10, -7, -4, -1, 2]]))
        np.testing.assert_array_equal(traceback, np.array([[0, 3, 3, 3, 3],
                                                           [2, 1, 5, 3, 3],
                                                           [2, 4, 1, 3, 3],
                                                           [2, 2, 2, 1, 5],
                                                           [2, 2, 2, 1, 1],
                                                           [2, 2, 2, 2, 1]]))

    def test_compute_needleman_wunsch_2(self):
        seq1 = "CIAO"
        seq2 = "CIAOCI"
        matrix, traceback = compute_needleman_wunsch(2, -1, -2, seq1, seq2, 1)
        np.testing.assert_array_equal(matrix, np.array([[0, -2, -4, -6, -8, -10, -12],
                                                        [-2, 2, 0, -2, -4, -6, -8],
                                                        [-4, 0, 4, 2, 0, -2, -4],
                                                        [-6, -2, 2, 6, 4, 2, 0],
                                                        [-8, -4, 0, 4, 8, 6, 4]]))
        np.testing.assert_array_equal(traceback, np.array([[0, 3, 3, 3, 3, 3, 3],
                                                           [2, 1, 3, 3, 3, 5, 3],
                                                           [2, 2, 1, 3, 3, 3, 5],
                                                           [2, 2, 2, 1, 3, 3, 3],
                                                           [2, 2, 2, 2, 1, 3, 3]]))

    def test_compute_needleman_wunsch_3(self):
        seq1 = "ACACACC"
        seq2 = "ACA"
        matrix, traceback = compute_needleman_wunsch(3, -1, -1, seq1, seq2, 1)
        np.testing.assert_array_equal(matrix, np.array([[0, -1, -2, -3],
                                                        [-1, 3, 2, 1],
                                                        [-2, 2, 6, 5],
                                                        [-3, 1, 5, 9],
                                                        [-4, 0, 4, 8],
                                                        [-5, -1, 3, 7],
                                                        [-6, -2, 2, 6],
                                                        [-7, -3, 1, 5]]))
        np.testing.assert_array_equal(traceback, np.array([[0, 3, 3, 3],
                                                           [2, 1, 3, 5],
                                                           [2, 2, 1, 3],
                                                           [2, 4, 2, 1],
                                                           [2, 2, 4, 2],
                                                           [2, 4, 2, 4],
                                                           [2, 2, 4, 2],
                                                           [2, 2, 4, 2]]))

    def test_compute_needleman_wunsch_4(self):
        seq1 = "AATCG"
        seq2 = "AACG"
        matrix, traceback = compute_needleman_wunsch(1, -1, -2, seq1, seq2, 0)
        np.testing.assert_array_equal(matrix, np.array([[0, -2, -4, -6, -8],
                                                        [-2, 1, -1, -3, -5],
                                                        [-4, -1, 2, 0, -2],
                                                        [-6, -3, 0, 1, -1],
                                                        [-8, -5, -2, 1, 0],
                                                        [-10, -7, -4, -1, 2]]))
        np.testing.assert_array_equal(traceback, np.array([[0, 3, 3, 3, 3],
                                                           [2, 1, 1, 3, 3],
                                                           [2, 1, 1, 3, 3],
                                                           [2, 2, 2, 1, 1],
                                                           [2, 2, 2, 1, 1],
                                                           [2, 2, 2, 2, 1]]))

    def test_compute_needleman_wunsch_5(self):
        seq1 = "TGCT"
        seq2 = "ATTCA"
        matrix, traceback = compute_needleman_wunsch(3, -1, -3, seq1, seq2, 1)
        np.testing.assert_array_equal(matrix, np.array([[0, -3, -6, -9, -12, -15],
                                                        [-3, -1, 0, -3, -6, -9],
                                                        [-6, -4, -2, -1, -4, -7],
                                                        [-9, -7, -5, -3, 2, -1],
                                                        [-12, -10, -4, -2, -1, 1]]))
        np.testing.assert_array_equal(traceback, np.array([[0, 3, 3, 3, 3, 3],
                                                           [2, 1, 1, 5, 3, 3],
                                                           [2, 4, 1, 1, 5, 5],
                                                           [2, 4, 4, 1, 1, 3],
                                                           [2, 4, 1, 1, 2, 1]]))

    def test_compute_needleman_wunsch_6(self):
        seq1 = "TGCT"
        seq2 = "ATTCA"
        matrix, traceback = compute_needleman_wunsch(3, -1, -3, seq1, seq2, 0)
        np.testing.assert_array_equal(matrix, np.array([[0, -3, -6, -9, -12, -15],
                                                        [-3, -1, 0, -3, -6, -9],
                                                        [-6, -4, -2, -1, -4, -7],
                                                        [-9, -7, -5, -3, 2, -1],
                                                        [-12, -10, -4, -2, -1, 1]]))
        np.testing.assert_array_equal(traceback, np.array([[0, 3, 3, 3, 3, 3],
                                                           [2, 1, 1, 1, 3, 3],
                                                           [2, 1, 1, 1, 1, 1],
                                                           [2, 1, 1, 1, 1, 3],
                                                           [2, 1, 1, 1, 2, 1]]))

    @patch('sys.stdout', new_callable=io.StringIO)
    def test_traceback_smith_waterman_1(self, mock_stdout):
        seq1 = "AATCG"
        seq2 = "AACG"
        matrix, traceback = compute_smith_waterman(1, -1, -2, seq1, seq2, 1)
        traceback_smith_waterman(seq1, seq2, matrix, traceback, 1)
        self.assertEqual(mock_stdout.getvalue(), "Alignment with score 2:\nAA\nAA\nAlignment with score 2:\nCG\nCG\n")

    @patch('sys.stdout', new_callable=io.StringIO)
    def test_traceback_smith_waterman_2(self, mock_stdout):
        seq1 = "CIAO"
        seq2 = "CIAOCI"
        matrix, traceback = compute_smith_waterman(2, -1, -2, seq1, seq2, 1)
        traceback_smith_waterman(seq1, seq2, matrix, traceback, 1)
        self.assertEqual(mock_stdout.getvalue(), "Alignment with score 8:\nCIAO\nCIAO\n")

    @patch('sys.stdout', new_callable=io.StringIO)
    def test_traceback_smith_waterman_3(self, mock_stdout):
        seq1 = "ACACACC"
        seq2 = "ACA"
        matrix, traceback = compute_smith_waterman(3, -1, -1, seq1, seq2, 1)
        traceback_smith_waterman(seq1, seq2, matrix, traceback, 1)
        self.assertEqual(mock_stdout.getvalue(), "Alignment with score 9:\nACA\nACA\nAlignment with score 9:\nACA\nACA\n")

    @patch('sys.stdout', new_callable=io.StringIO)
    def test_traceback_smith_waterman_4(self, mock_stdout):
        seq1 = "AATCG"
        seq2 = "AACG"
        matrix, traceback = compute_smith_waterman(1, -1, -2, seq1, seq2, 0)
        traceback_smith_waterman(seq1, seq2, matrix, traceback, 0)
        self.assertEqual(mock_stdout.getvalue(), "Alignment with score 2:\nAA\nAA\n")

    @patch('sys.stdout', new_callable=io.StringIO)
    def test_traceback_smith_waterman_5(self, mock_stdout):
        seq1 = "TGCT"
        seq2 = "ATTCA"
        matrix, traceback = compute_smith_waterman(3, -1, -3, seq1, seq2, 0)
        traceback_smith_waterman(seq1, seq2, matrix, traceback, 1)
        self.assertEqual(mock_stdout.getvalue(), "Alignment with score 5:\nTGC\nTTC\n")

    @patch('sys.stdout', new_callable=io.StringIO)
    def test_traceback_smith_waterman_6(self, mock_stdout):
        seq1 = "TGCT"
        seq2 = "ATTCA"
        matrix, traceback = compute_smith_waterman(3, -1, -3, seq1, seq2, 1)
        traceback_smith_waterman(seq1, seq2, matrix, traceback, 1)
        self.assertEqual(mock_stdout.getvalue(), "Alignment with score 5:\nTGC\nTTC\n")

    @patch('sys.stdout', new_callable=io.StringIO)
    def test_traceback_needleman_wunsch_1(self, mock_stdout):
        seq1 = "AATCG"
        seq2 = "AACG"
        matrix, traceback = compute_needleman_wunsch(1, -1, -2, seq1, seq2, 1)
        path(seq1, seq2, traceback, len(seq1), len(seq2), matrix[len(seq1), len(seq2)])
        self.assertEqual(mock_stdout.getvalue(), "Alignment with score 2:\nAATCG\nAA-CG\n")

    @patch('sys.stdout', new_callable=io.StringIO)
    def test_traceback_needleman_wunsch_2(self, mock_stdout):
        seq1 = "CIAO"
        seq2 = "CIAOCI"
        matrix, traceback = compute_needleman_wunsch(2, -1, -2, seq1, seq2, 1)
        path(seq1, seq2, traceback, len(seq1), len(seq2), matrix[len(seq1), len(seq2)])
        self.assertEqual(mock_stdout.getvalue(), "Alignment with score 4:\nCIAO--\nCIAOCI\n")

    @patch('sys.stdout', new_callable=io.StringIO)
    def test_traceback_needleman_wunsch_3(self, mock_stdout):
        seq1 = "ACACACC"
        seq2 = "ACA"
        matrix, traceback = compute_needleman_wunsch(3, -1, -1, seq1, seq2, 1)
        path(seq1, seq2, traceback, len(seq1), len(seq2), matrix[len(seq1), len(seq2)])
        self.assertEqual(mock_stdout.getvalue(),
                         "Alignment with score 5:\nACACACC\n--ACA--\n"
                         "Alignment with score 5:\nACACACC\nACA----\n"
                         "Alignment with score 5:\nACACACC\nAC--A--\n"
                         "Alignment with score 5:\nACACACC\nA--CA--\n")

    @patch('sys.stdout', new_callable=io.StringIO)
    def test_traceback_needleman_wunsch_4(self, mock_stdout):
        seq1 = "ACACACC"
        seq2 = "ACA"
        matrix, traceback = compute_needleman_wunsch(3, -1, -1, seq1, seq2, 0)
        path(seq1, seq2, traceback, len(seq1), len(seq2), matrix[len(seq1), len(seq2)])
        self.assertEqual(mock_stdout.getvalue(),
                         "Alignment with score 5:\nACACACC\n--ACA--\n")

    @patch('sys.stdout', new_callable=io.StringIO)
    def test_traceback_needleman_wunsch_5(self, mock_stdout):
        seq1 = "ATTAAAGGTTTATACCTTCCCAGGTAACAAACCAACCAACTTTCGATCTCTTGTAGATCTGTTCTCTAAACGAACTTTAAAATCTGTGTGGCTGTCACTCGGCTGCATGCTTAGTGCACTCACGCAGTATAATTAATAACTAATTACTGTCGTTGACAGGACACGAGTAACTCGTCTATCTTCTGCAGGCTGCTTACGGTTTCGTCCGTGTTGCAGCCGATCATCAGCACATCTAGGTTTCGTCCGGGTGTGACCGAAAGGTAAGATGGAGAGCCTTGTCCCTGGTTTCAACGAGAAAACACACGTCCAACTCAGTTTGCCTGTTTTACAGGTTCGCGACGTGCTCGTACGTGGCTTTGGAGACTCCGTGGAGGAGGTCTTATCAGAGGCACGTCAACATCTTAAAGATGGCACTTGTGGCTTAGTAGAAGTTGAAAAAGGCGTTTTGCCTCAACTTGAACAGCCCTATGTGTTCATCAAACGTTCGGATGCTCGAACTGCACCTCATGGTCATGT"
        seq2 = "TATGGTTGAGCTGGTAGCAGAACTCGAAGGCATTCAGTACGGTCGTAGTGGTGAGACACTTGGTGTCCTTGTCCCTCATGTGGGCGTAATACCAGTGGCTTACCGCAAGGTTCTTCTTCGTAAGAACGGTAATAAAGGAGCTGGTGGCCATAGTTACGGCGCCGATCTAAAGTCATTTGACTTAGGCGACGAGCTTGGCACTGATCCTTATGAAGATTTTCAAGAAAACTGGAACACTAAACATAGCAGTGGTGTTACCCGTGAACTCATGCGTGAGCTTAACGGAGGGGCATACACTCGCTATGTCGATAACAACTTCTGTGGCCCTGATGGCTACCCTCTTGAGTGCATTAAAGACCTTCTAGCACGTGCTGGTAAAGCTTCATGCACTTTGTCCGAACAACTGGACTTTATTGACACTAAGAGGGGTGTATACTGCTGCCGTGAACATGAGCATGAAATTG"
        matrix, traceback = compute_needleman_wunsch(3, -1, -1, seq1, seq2, 0)
        path(seq1, seq2, traceback, len(seq1), len(seq2), matrix[len(seq1), len(seq2)])
        self.assertEqual(mock_stdout.getvalue(), "Alignment with score 679:\n"
                                                 "ATTAAAGGTTTATACCTTCCCAGGTAACA-AAC-C-AA-CCAACTTTC-G-ATC--TCTTGTAGATCTGTTCTCTAAACGAACTTTAAAATCTGTGTGGC-TGTCACTCGGCTGCATGCTTAGTGC--ACTCACGCAGTATAATTAATAACTAA--TTACTGTCGTT-G-ACAGGACACGAGTAACTCGTCTATCTTCTGCAGGCTGCTTACGGTTTCGTCCGT-GTT---GCAGCCGATCATCAGCACA-TCTAGGTTTCGTCCGGGTGTGACCGAAAGGTAAGATGG-A--GAGCCTTGTCCCTG--G--TTTCAACGAGAAA----ACACACGTCCAAC-T--CAGTTTGCCTGTTTTACAGGTTCGCGACGTGC-TCGTACGTG-GCTTTGGAGACTCCGTGGAGGAGGTCTTATCAGAGGC-A---CG-T--CAACATCT-T---AAAGATGGC-A----CTTGTG-GC-TT--AG------TAG-AAGT--TGAAAAAGGCGTT-TTGC-C----T-C-AACTTGAAC-AG-CCCTATGTGTTCA-TCAA-ACGTTCGGATGCTCGA-ACTGC-ACC-T---CATG-GTCATG----T-\n"
                                                 "--T-ATGG-TT-GAGC-T----GGTAGCAGAACTCGAAGGC-A--TTCAGTA-CGGTC--GTAG---TG--GT-GAGAC--AC--T----T-GGTGT-CCTTGT--C-C--CT-CATG--T-GGGCGTAAT-AC-CAGT-GGCTT-ACCGC-AAGGTT-CT-TC-TTCGTA-A-GA-ACG-GTAA------TA-------AAGG-AGC-T--GG--T-GGCCATAGTTACGGC-GCCGATC-T-A--A-AGTC-A--TTT-GACTTAG-GCGA-CG--AGCT----TGGCACTGATCC-T-T--ATGAAGATTTTCAA-GA-AAACTGGA-ACAC-T-AAACATAGCAG--TG---GTGTTACCCG-T-G-AAC-T-CAT-G--CGTGAGC-TT--A-A---C--GGAGG-GG-CATA-CACTCGCTATGTCGATAACAACTTCTGTGGCCCTGATGGCTACCCTCTTGAGTGCATTAAAGACCTTCTAGCACGTGCTGGTAAA-GC-TTCATGCACTTTGTCCGAAC---AACTGGACTTTAT-TG-ACACT-AAGA-G---GGGTG-T--ATACTGCTGCCGTGAACATGAG-CATGAAATTG\n")

    @patch('sys.stdout', new_callable=io.StringIO)
    def test_traceback_needleman_wunsch_6(self, mock_stdout):
        seq1 = "ATTCA"
        seq2 = "TGCT"
        matrix, traceback = compute_needleman_wunsch(3, -1, -3, seq1, seq2, 1)
        path(seq1, seq2, traceback, len(seq1), len(seq2), matrix[len(seq1), len(seq2)])
        self.assertEqual(mock_stdout.getvalue(),
                         "Alignment with score 1:\nATTCA\n-TGCT\n")

    @patch('sys.stdout', new_callable=io.StringIO)
    def test_traceback_needleman_wunsch_7(self, mock_stdout):
        seq1 = "ATTCA"
        seq2 = "TGCT"
        matrix, traceback = compute_needleman_wunsch(3, -1, -3, seq1, seq2, 0)
        path(seq1, seq2, traceback, len(seq1), len(seq2), matrix[len(seq1), len(seq2)])
        self.assertEqual(mock_stdout.getvalue(),
                         "Alignment with score 1:\nATTCA\n-TGCT\n")


if __name__ == '__main__':
    unittest.main()
