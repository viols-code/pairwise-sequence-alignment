# Project name: Pairwise sequence alignment

# Programming language: Python

# Author: Viola Renne

# Short description: Implement a pairwise sequence alignment method, to find the optimal
# global sequence alignment between two nucleotide sequences, using both Smith-Waterman and Needleman–Wunsch
# algorithm.
# Expected outcome:
#   Develop a script which consist of:
#       1. Appropriate data structures;
#       2. Based on the design in the point (1), implement the two algorithms for optimal local and global alignment.
#       3. The expected outcome of the project is a function that takes as parameters: (a1) the cost of a
#           match, (a2) the cost of a mismatch, (a3) the cost of a gap/indel, (b) the first nucleotide
#           sequence, (c) the second nucleotide sequence. As result, returns one optimal alignment
#           between the two input sequences (b) and (c).

from optparse import OptionParser
import numpy as np
from enum import Enum


class Step(Enum):
    """
    Enumeration for the step in the traceback
    """
    STOP = 0
    DIAG = 1
    UP = 2
    LEFT = 3
    DIAG_UP = 4
    DIAG_LEFT = 5
    UP_LEFT = 6
    DIAG_UP_LEFT = 7


def determine_step(prev_value, diag, up, left):
    """
    Determine step for traceback
    :param prev_value: position of the max value
    :param diag: score obtain from the diagonal cell
    :param up: score obtain from the up cell
    :param left: score obtain from the left cell
    """
    if prev_value == 0:
        return Step.STOP.value
    elif prev_value == 1:
        if diag == left and diag == up:
            return Step.DIAG_UP_LEFT.value
        elif diag == up:
            return Step.DIAG_UP.value
        elif diag == left:
            return Step.DIAG_LEFT.value
        else:
            return Step.DIAG.value
    elif prev_value == 2:
        if left == up:
            return Step.UP_LEFT.value
        else:
            return Step.UP.value
    elif prev_value == 3:
        return Step.LEFT.value


def compute_needleman_wunsch(m, s, g, sequence1, sequence2, al):
    """
    Compute procedure matrix and traceback matrix for the Needleman-Wunsch algorithm
    :param m: match score
    :param s: mismatch score
    :param g: gap score
    :param sequence1: first sequence
    :param sequence2: second sequence
    :param al: 0 if only one optimal alignment need to be returned, 1 otherwise
    """
    len1 = len(sequence1)
    len2 = len(sequence2)
    procedure_matrix = np.zeros((len1 + 1, len2 + 1), dtype=int)
    traceback_matrix = np.zeros((len1 + 1, len2 + 1), dtype=int)

    # Initialization
    for i in range(len1):
        procedure_matrix[i + 1, 0] = (i + 1) * g
        traceback_matrix[i + 1, 0] = Step.UP.value
    for j in range(len2):
        procedure_matrix[0, j + 1] = (j + 1) * g
        traceback_matrix[0, j + 1] = Step.LEFT.value

    # Compute the procedure and traceback matrices
    for i in range(len1):
        for j in range(len2):
            sigma = m if sequence1[i] == sequence2[j] else s
            procedure_matrix[i + 1, j + 1] = np.max((procedure_matrix[i, j] + sigma, procedure_matrix[i, j + 1] + g,
                                                     procedure_matrix[i + 1, j] + g))
            traceback_matrix[i + 1, j + 1] = np.argmax((procedure_matrix[i, j] + sigma, procedure_matrix[i, j + 1] + g,
                                                        procedure_matrix[i + 1, j] + g,)) + 1
            if al:
                traceback_matrix[i + 1, j + 1] = determine_step(traceback_matrix[i + 1, j + 1],
                                                                procedure_matrix[i, j] + sigma,
                                                                procedure_matrix[i, j + 1] + g,
                                                                procedure_matrix[i + 1, j] + g)

    return procedure_matrix, traceback_matrix


def compute_smith_waterman(m, s, g, sequence1, sequence2, al):
    """
    Compute procedure matrix and traceback matrix for the Smith-Waterman algorithm
    :param m: match score
    :param s: mismatch score
    :param g: gap score
    :param sequence1: first sequence
    :param sequence2: second sequence
    :param al: 0 if only one optimal alignment need to be returned, 1 otherwise
    """
    len1 = len(sequence1)
    len2 = len(sequence2)
    procedure_matrix = np.zeros((len1 + 1, len2 + 1), dtype=int)
    traceback_matrix = np.zeros((len1 + 1, len2 + 1), dtype=int)
    # Compute the procedure and traceback matrices
    for i in range(len1):
        for j in range(len2):
            sigma = m if sequence1[i] == sequence2[j] else s
            procedure_matrix[i + 1, j + 1] = np.max((0, procedure_matrix[i, j] + sigma, procedure_matrix[i, j + 1] + g,
                                                     procedure_matrix[i + 1, j] + g))
            traceback_matrix[i + 1, j + 1] = np.argmax((0, procedure_matrix[i, j] + sigma,
                                                        procedure_matrix[i, j + 1] + g, procedure_matrix[i + 1, j] + g))
            if al:
                traceback_matrix[i + 1, j + 1] = determine_step(traceback_matrix[i + 1, j + 1],
                                                                procedure_matrix[i, j] + sigma,
                                                                procedure_matrix[i, j + 1] + g,
                                                                procedure_matrix[i + 1, j] + g)

    return procedure_matrix, traceback_matrix


def path(sequence1, sequence2, traceback_matrix, r, c):
    """
    Compute alignments from the traceback matrix
    :param sequence1: first sequence
    :param sequence2: second sequence
    :param traceback_matrix: traceback matrix
    :param r: row position where the alignment start
    :param c: column position where the alignment start
    """
    # Initializing the first alignment
    alignments = [['', '', r, c]]

    while len(alignments) != 0:  # As long as there are alignments
        flag = 0
        alignment = alignments[0]
        s = alignment[0]
        t = alignment[1]
        r = alignment[2]
        c = alignment[3]
        if traceback_matrix[r, c] == Step.STOP.value:  # End of alignment
            print("Alignment:")
            print(alignment[0])
            print(alignment[1])
            alignments.remove(alignment)
        else:  # Traceback step
            if traceback_matrix[r, c] in [Step.DIAG.value, Step.DIAG_UP.value, Step.DIAG_LEFT.value,
                                          Step.DIAG_UP_LEFT.value]:  # Traceback step is diagonal
                alignments[0] = [sequence1[r - 1] + s, sequence2[c - 1] + t, r - 1, c - 1]
                flag = 1
            if traceback_matrix[r, c] in [Step.LEFT.value, Step.DIAG_LEFT.value, Step.UP_LEFT.value,
                                          Step.DIAG_UP_LEFT.value]:  # Traceback step is left
                if flag:
                    alignments.append(['-' + s, sequence2[c - 1] + t, r, c - 1])
                else:
                    alignments[0] = ['-' + s, sequence2[c - 1] + t, r, c - 1]
                    flag = 1
            if traceback_matrix[r, c] in [Step.UP.value, Step.DIAG_UP.value, Step.UP_LEFT.value,
                                          Step.DIAG_UP_LEFT.value]:  # Traceback step is up
                if flag:
                    alignments.append([sequence1[r - 1] + s, '-' + t, r - 1, c])
                else:
                    alignments[0] = [sequence1[r - 1] + s, '-' + t, r - 1, c]


def traceback_smith_waterman(sequence1, sequence2, procedure_matrix, traceback_matrix, al):
    """
    Find the row and column coordinates of the cells with the maximum score
    :param sequence1: first sequence
    :param sequence2: second sequence
    :param procedure_matrix: procedure matrix
    :param traceback_matrix: traceback matrix
    :param al: 0 if only one optimal alignment need to be returned, 1 otherwise
    """
    max_xy = np.where(procedure_matrix == np.max(procedure_matrix))
    x = max_xy[0]
    y = max_xy[1]
    for i in range(len(x)):
        path(sequence1, sequence2, traceback_matrix, x[i], y[i])
        if al == 0:
            return


if __name__ == '__main__':
    parser = OptionParser()
    """ Adding all the options that can be given as parameters """
    parser.add_option("-m", action="store", type="int", dest="match", help="cost of a match")
    parser.add_option("-s", action="store", type="int", dest="mismatch", help="cost of the mismatch")
    parser.add_option("-i", action="store", type="int", dest="indel", help="cost of the gap/indel")
    parser.add_option("-b", action="store", type="string", dest="seq1", help="first sequence")
    parser.add_option("-c", action="store", type="string", dest="seq2", help="second sequence")
    parser.add_option("-a", action="store", type="string", dest="algorithm", help="Type 'local' for local alignment "
                                                                                  "and 'global' for global alignment")
    parser.add_option("-t", action="store", type="int", dest="alignment", help="Type '0' for only one alignment "
                                                                               "and '1' for all possible alignments")

    """ Reading parameters """
    (options, args) = parser.parse_args()
    match = options.match
    mismatch = options.mismatch
    gap = options.indel
    seq1 = options.seq1
    seq2 = options.seq2
    algorithm = options.algorithm
    align = options.alignment

    if algorithm == 'local':
        matrix, traceback = compute_smith_waterman(match, mismatch, gap, seq1, seq2, align)
        traceback_smith_waterman(seq1, seq2, matrix, traceback, align)
    elif algorithm == 'global':
        matrix, traceback = compute_needleman_wunsch(match, mismatch, gap, seq1, seq2, align)
        path(seq1, seq2, traceback, len(seq1), len(seq2))
    else:
        print("Insert 'local' for local alignment and 'global' for global alignment")
