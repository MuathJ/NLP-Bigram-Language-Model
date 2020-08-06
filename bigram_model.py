##########-------------- Muath Juady - 11440920 --------------##########

import glob
import time
import nltk
from nltk.stem import PorterStemmer
from nltk.tokenize import sent_tokenize, word_tokenize

#Choice = int(input("Training (0) OR Testing (1) : "))
#if (Choice==0):

############################################# TRAINING ##############################################

#--------------- Convert All files into 1 txt (4.4 sec) ----------------
start = time.time()
print ('\nConverting All files into training.txt ...')

with open("training.txt", "wb") as outfile:
    for f in glob.glob("./training/*/*", recursive=True):
        with open(f, "rb") as infile:
            outfile.write(infile.read())

end = time.time()
print ('Converting All files into training.txt ... DONE IN '+ str(round(end-start, 2)) +' Seconds \n')
#-----------------------------------------------------------------------



#---------------- Unigram List + Tokenization + Stemming ---------------
start = time.time()
print ('Generating Unigram List + Tokenization + Stemming...')

UniList = []
with open("training.txt", "r") as training:
    for word in word_tokenize(training.read()):
        UniList.append(PorterStemmer().stem(word))

end = time.time()
print ('Generating Unigram List + Tokenization + Stemming... DONE IN '+ str(round(end-start, 2)) +' Seconds')
print ('## Result: ', len(UniList), ' Unigram Tokens\n')
#-----------------------------------------------------------------------



#-------------- Unigram/Bigram Frequencies + Bigram List ---------------
start = time.time()
print ('Generating Unigram/Bigram Frequencies + Bigram List...')

BiList = []
UniCount = {}
BiCount = {}

for i in range(len(UniList)):

    # Generate Unigrams Frequencies
    # -------------------------------
    if UniList[i] in UniCount:
        UniCount[UniList[i]] += 1
    else:
        UniCount[UniList[i]] = 1
    # -------------------------------

    # Generate Bigram Frequencies + Bigram List
    # ------------------------------------------------
    if i < len(UniList) - 1:
        BiList.append( (UniList[i], UniList[i+1]) )

        if (UniList[i], UniList[i+1]) in BiCount:
            BiCount[(UniList[i], UniList[i+1])] += 1
        else:
            BiCount[(UniList[i], UniList[i+1])] = 1
    # ------------------------------------------------

#with open("UniCount.txt", "w") as Uni:
#    Uni.write(str(UniCount))

end = time.time()
print ('Generating Unigram/Bigram Frequencies + Bigram List... DONE IN '+ str(round(end-start, 2)) +' Seconds')
print ('## Result: ', len(BiList), ' Bigram Tokens')
print ('## Result: ', len(UniCount), ' Unique Unigrams (V)')
print ('## Result: ', len(BiCount), ' Unique Bigrams \n')
#-----------------------------------------------------------------------



#--------- Bigram Model Prob + Add-1 Laplacian Smoothing (Reconstituted Counts) ---------
start = time.time()
print ('Generating Bigram Model Prob File + Add-1 Laplacian Smoothing...')

with open("Bigram-Prob.txt", "w") as file:
    file.write('Bigram' + '\t\t\t' + 'Count' + '\t' + 'Probability' + '\n')
    ProbList = {}
    for bigram in BiList:
        ProbList[bigram] = (BiCount[bigram] + 1) * UniCount[bigram[0]] / (UniCount[bigram[0]] + len(UniCount))
        file.write(str(bigram) + ' ## ' + str(BiCount[bigram]) + ' ## ' + str(ProbList[bigram]) + '\n')

end = time.time()
print ('Generating Bigram Model Prob File + Add-1 Laplacian Smoothing... DONE IN '+ str(round(end-start, 2)) +' Seconds \n')
#----------------------------------------------------------------------------------------


########################################## END OF TRAINING ##########################################


#if (Choice==1):
############################################## TESTING ##############################################

#---------------------------- Entering Sentences For Testing --------------------------------


TProbFile = open('Testing-Prob.txt', 'w')

with open("test.txt", "r") as f:
    lines = f.readlines()
    for input in lines:
        StemTok = []
        InputList = []
        TProb = 1

        for word in word_tokenize(input):
            StemTok.append(PorterStemmer().stem(word))

        for i in range(len(StemTok)-1):
            InputList.append( (StemTok[i], StemTok[i+1])  )
        #----------------------------------------------------------------------------------------

        # -------------------------------- Bigram Model Testing ---------------------------------

        #with open("UniCount.txt", "r") as UniCount:
        with open("Bigram-Prob.txt", "r") as ProbListFile:
            for i in range(len(InputList)):
                if InputList[i] in ProbList:
                    TProb *= ProbList[InputList[i]]
                else:
                    if InputList[i][0] not in UniCount:
                        UniCount[InputList[i][0]] = 1
                    Prob = (1) / (UniCount[InputList[i][0]] + len(UniCount))
                    TProb *= Prob

            TProbFile.write('\n\n' + input + '\t' + 'Probablility = ' + str(TProb))
# ------------------------------- --------- ---------------------------------------------------

########################################### END OF TESTING ###########################################

#else:
#    exit( 0 )
