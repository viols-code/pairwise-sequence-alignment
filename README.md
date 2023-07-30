# Pairwise sequence alignment

## :bookmark_tabs: Menu
* [Overview](#overview)
* [Description](#description)
* [Usage](#usage)

## Overview
This is a project proposed by the Scientific Programming course by Politecnico di Milano

## Description
**Project name:** Pairwise sequence alignment
**Programming language:** Python  
**Short description:** Implement a pairwise global sequence alignment method, to find the optimal
global sequence alignment between two nucleotide sequences, using both Smith-Waterman and Needleman–Wunsch algorithms.  
**Expected outcome:** 
Develop a script which consist of:
1. Appropriate data structures;
2. Based on the design in the point (1), implement the two algorithms for optimal alignment.
3. The expected outcome of the project is a function that takes as parameters: (a1) the cost of a
match, (a2) the cost of a mismatch, (a3) the cost of a gap/indel, (b) the first nucleotide
sequence, (c) the second nucleotide sequence. As result, returns one optimal alignment
between the two input sequences (b) and (c).

## Usage

1. **Install the requirements.**  
    In order to install the requirements, use:  
    ```bash
    pip install -r requirements.txt
    ```
   
2. **Add options.**  
In order to see the possible options, open a terminal, cd into project directory and type:
   ```bash
   python sequence_alignment.py --help 
   ```

3. **Standard simulation.**
Open a terminal into project directory and type:
   ```bash
   python sequence_alignment.py -m match_cost -s mismatch_cost -i indel_cost -b first_sequence -c second_sequence -a algorithm -t alignments
   ```
   
4. **Tests.**
Open a terminal into project directory and type:
   ```bash
    python -m unittest -v
   ```