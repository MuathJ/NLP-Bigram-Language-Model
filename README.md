# NLP Bigram Language Model

Build a bi-gram language model whose parameters are trained from a corpus of documents. This corpus contains 91 folders, where each has a set of documents.
Follow these steps that help in building a good LM:

1.	Word Tokenization: splitting the text into uni-gram tokens.
You can either use your own tokenization methodology or the Stanford tokenizer: http://nlp.stanford.edu/software/tokenizer.html:


2.	Token normalization: use Porters’ Stemmer to return a token back to its base.

3.	Vocabulary set extraction.

4.	Estimation of model parameters, i.e., p(wi|wi-1)

*Do not forget to handle Zero probabilities using Add-1 Laplacian Smoothing*
*Do not forget to handle unknown words, words appearing at testing time.*
*Save the parameters of the bi-gram model (probabilities) in a file, so that, at testing time, you need only to load such parameters and not to learn them again.*

5.	At testing time, we give your model a sentence, and it calculates its probability.
