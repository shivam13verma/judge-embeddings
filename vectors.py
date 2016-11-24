import locale
import glob
import os
import os.path
import requests
import tarfile
import sys
import re

import gensim
from gensim.models.doc2vec import TaggedDocument
from collections import namedtuple

from gensim.models import Doc2Vec
import gensim.models.doc2vec
from collections import OrderedDict
import multiprocessing

from gensim.test.test_doc2vec import ConcatenatedDoc2Vec

import pickle


reload(sys)
sys.setdefaultencoding("utf-8")


dirname = '/scratch/ap4608/judge_data'
locale.setlocale(locale.LC_ALL, 'C')


# Convert text to lower-case and strip punctuation/symbols from words
def normalize_text(text):
    norm_text = text.lower()

    # Replace breaks with spaces
    norm_text = norm_text.replace('<br />', ' ')

    # Pad punctuation with spaces on both sides
    for char in ['.', '"', ',', '(', ')', '!', '?', ';', ':']:
        norm_text = norm_text.replace(char, ' ' + char + ' ')

    return norm_text


# Concat and normalize test/train data
folders = os.listdir(dirname)
alldata = ''

for fol in folders:
    temp = ''
    output = fol.replace('/', '-') + '.txt'

    # Is there a better pattern to use?
    txt_files = glob.glob('/'.join([dirname, fol, '*.txt']))

    for txt in txt_files:
        with open(txt, 'r') as t:
            control_chars = [chr(0x85)]
            t_clean = t.read()

            t_clean = t_clean.replace('\n', ' ')
            t_clean = re.sub(r'[^\x00-\x7F]+',' ', t_clean)

            for c in control_chars:
                t_clean = t_clean.replace(c, ' ')

            temp += t_clean

    temp += "\n"

    temp_norm = normalize_text(temp)

    if len(temp_norm) == 1:
        continue

    with open('/'.join([dirname, output]), 'w') as n:
        n.write(temp_norm)

    alldata += temp_norm

with open('/'.join([dirname, 'alldata-id.txt']), 'w') as f:
    for idx, line in enumerate(alldata.splitlines()):
        num_line = "_*{0} {1}\n".format(idx, line)
        f.write(num_line)

SentimentDocument = namedtuple('SentimentDocument', 'words tags split sentiment')

alldocs = []  # will hold all docs in original order
with open(os.path.join(dirname, 'alldata-id.txt')) as alldata:
    for line_no, line in enumerate(alldata):
        tokens = gensim.utils.to_unicode(line).split()
        words = tokens[1:]
        tags = [line_no] # `tags = [tokens[0]]` would also work at extra memory cost
        split = ['train','test','extra','extra'][line_no//25000]  # 25k train, 25k test, 25k extra
        sentiment = [1.0, 0.0, 1.0, 0.0, None, None, None, None][line_no//12500] # [12.5K pos, 12.5K neg]*2 then unknown
        alldocs.append(SentimentDocument(words, tags, split, sentiment))

train_docs = [doc for doc in alldocs if doc.split == 'train']
test_docs = [doc for doc in alldocs if doc.split == 'test']
doc_list = alldocs[:]  # for reshuffling per pass

cores = multiprocessing.cpu_count()
assert gensim.models.doc2vec.FAST_VERSION > -1, "this will be painfully slow otherwise"

simple_models = [
    # PV-DM w/concatenation - window=5 (both sides) approximates paper's 10-word total window size
    Doc2Vec(dm=1, dm_concat=1, size=100, window=5, negative=5, hs=0, min_count=2, workers=cores),
    # PV-DBOW 
    Doc2Vec(dm=0, size=100, negative=5, hs=0, min_count=2, workers=cores),
    # PV-DM w/average
    Doc2Vec(dm=1, dm_mean=1, size=100, window=10, negative=5, hs=0, min_count=2, workers=cores),
]

# speed setup by sharing results of 1st model's vocabulary scan
simple_models[0].build_vocab(alldocs)  # PV-DM/concat requires one special NULL word so it serves as template
for model in simple_models[1:]:
    model.reset_from(simple_models[0])

models_by_name = OrderedDict((str(model), model) for model in simple_models)

models_by_name['dbow+dmm'] = ConcatenatedDoc2Vec([simple_models[1], simple_models[2]])
models_by_name['dbow+dmc'] = ConcatenatedDoc2Vec([simple_models[1], simple_models[0]])

# Create a document vector list and save it
doc_vec_list = [x.docvecs for x in simple_models]
pickle.dump(doc_vec_list, open('docvecs.p', 'wb'))

# pickle.dump(models_by_name, open('model.p', 'wb'))
