import os, sys
import string, re

folder_path = "/scratch/sv1239/projects/chen/raw/"  #path to Archive folder containing folders for all years (1880, 1881 etc.)
print [folder for folder in os.listdir(folder_path)]

def read_data(folder_path):
    corpus = "" #saves all documents into one
    for folder in os.listdir(folder_path):
        for file_name in os.listdir(folder_path+"/"+folder+"/"):
            if ".txt" in file_name:
                with open(folder_path+folder+"/"+file_name, 'r') as f:
                    corpus+=f.read()
        
    print corpus[:10]
    return corpus


# os.chdir(folder_path)\n\ncorpus = read_data(folder_path)')
print len(corpus)
print corpus[:1000]

text = corpus.translate(None, string.punctuation).replace("\n", " ") #remove punctuations, \n's
text = re.sub(r'[^\x00-\x7F]+',' ', text) #remove non-ASCII chars
text = re.sub( '\s+', ' ', text).lstrip().rstrip() #remove extra and trailing spaces

print text[:10000]

print len(text.split())

with open('circuit_corpus_1880_1890.txt', 'w') as f:
    f.write(text)

#get_ipython().system(u'wc -w circuit_corpus_1880_1890.txt')


# ###TODO
# 
# - Remove unicode chars - DONE
# - Replace numericals/dates with NN/NN/NNNN or NN.NNNN etc. - TBD. Check how people typically do it. Can also just remove all numbers.
# - GloVe vectors for 10 years (1880-1890) - DONE
# - GloVe vectors for entirety of dataset (~130 years) - TBD
# - Add bias tests comparing pre-trained and our GloVe vectors - TBD

# Note:
# 
# - While above process works for 10 years worth of data (32 MB), it may be a problem for larger datasets (> 100MB), since we are combining all the documents into one single document/string (separated by spaces), and creating a huge text file might lead to OOM issues. Can remedy by increasing memory in HPC (and specify RAM in run.sh below), but also look up how GloVe typically trained on huge corpuses, i.e. can we train in batches?
# - Once above process (corpus generation) is complete, do the following:
# - Go to https://github.com/stanfordnlp/GloVe and read the README
# - Clone the GloVe repo, try running the demo.sh
# - Copy the corpus file to the glove/ folder, and create a modified version of demo.sh, say, run.sh, which uses this file. Run.sh is provided in this repo.
# - Modify the params in run.sh accordingly (eg. dimension of vectors, threads etc.)
# 
# 




