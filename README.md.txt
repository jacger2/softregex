# SoftRegex: Generating Regex from Natural Language Descriptions using Softened Regex Equivalence

 This is implemantation of the paper [SoftRegex: Generating Regex from Natural Language Descriptions using Softened Regex Equivalence](https://www.aclweb.org/anthology/D19-1677/)
 
--------------

## Summary

We continue the study of generating semantically correct regular expressions from natural language descriptions (NL). The current state-of-the-art model SemRegex produces regular expressions from NLs by rewarding the reinforced learning based on the semantic (rather than syntactic) equivalence between two regular expressions. Since the regular expression equivalence problem is PSPACE-complete, we introduce the EQ_Reg model for computing the similarity of two regular expressions using deep neural networks. Our EQ_Reg model essentially softens the equivalence of two regular expressions when used as a reward function. We then propose a new regex generation model, SoftRegex, using the EQ_Reg model, and empirically demonstrate that SoftRegex substantially reduces the training time (by a factor of at least 3.6) and produces state-of-the-art results on three benchmark datasets.

--------------

### Install from source
    pip install -r requirements.txt
    python setup.py install

### train
    dataname = kb13, NL-RX-Synth, NL-RX-Turk
    python softregex-train.py (dataname)
    python softregex-eval.py (dataname)