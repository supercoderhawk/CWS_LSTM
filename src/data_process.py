# -*- coding: utf-8 -*-
import io
import os
import re
from shutil import copyfile
import numpy as np

training_filename_tmpl = '../sighan2005/processed_wo_idioms/{0}/{1}_train.utf8'
dest_training_filename_tmpl = ''
training_src_filename = training_filename_tmpl.format('raw', 'msr')
training_dest_filename = training_filename_tmpl.format('pro', 'msr')
test_filename_tmpl = '../sighan2005/processed_wo_idioms/{0}/{1}_test.utf8'
test_src_filename = test_filename_tmpl.format('raw', 'msr')
test_dest_filename = test_filename_tmpl.format('pro', 'msr')


def init_dir():
    if not os.path.exists('../sighan2005/'):
        os.mkdir('../sighan2005/')
    if not os.path.exists('../sighan2005/processed_wo_idioms/'):
        os.mkdir('../sighan2005/processed_wo_idioms/')
    if not os.path.exists(os.path.dirname(training_src_filename)):
        os.mkdir(os.path.dirname(training_src_filename))
    if not os.path.exists(os.path.dirname(training_dest_filename)):
        os.mkdir(os.path.dirname(training_dest_filename))
    if not os.path.exists('../PreTrainedWordEmbedding/'):
        os.mkdir('../PreTrainedWordEmbedding/')
    copyfile('../data/msr_train.utf8', training_src_filename)
    copyfile('../data/msr_test.utf8', test_src_filename)


def generate_dict_and_vectors():
    characters = {'<BOS>', '<EOS>', '<OOV>'}

    with io.open(training_src_filename, encoding='utf-8') as f:
        for line in f.read().splitlines():
            line = unicode(line)
            characters = characters.union(line.replace(' ', ''))

    with io.open(test_src_filename, encoding='utf-8') as f:
        for line in f.read().splitlines():
            line = unicode(line)
            characters = characters.union(line.replace(' ', ''))

    word_vectors = {}
    word_vector_len = 100
    for ch in characters:
        word_vectors[ch] = np.random.normal(0, 1, word_vector_len).tolist()

    with io.open('../PreTrainedWordEmbedding/charactor_OOVthr_50_%dv.txt' % 100, 'w', encoding='utf-8') as f:
        f.write(unicode(str(len(characters)) + ' ' + str(word_vector_len) + '\n'))
        for ch, vector in word_vectors.items():
            f.write(unicode(ch + ' ' + ' '.join(map(str, vector)) + '\n'))


def cws2conll(filename, dest_filename):
    with io.open(filename, encoding='utf-8') as f:
        conlls = []
        for line in f.readlines():
            line = line.strip()
            if not line:
                continue

            conll = []
            words = re.split(r'[ ]+', line)
            for word in words:
                word_len = len(word)
                if word_len == 1:
                    conll.append(word + ' ' + 'S')
                else:
                    conll.append(word[0] + ' ' + 'B')
                    if word_len > 2:
                        conll.extend([ch + ' ' + 'M' for ch in word[1:-1]])
                    conll.append(word[-1] + ' ' + 'E')
            conlls.append('\n'.join(conll))
        with io.open(dest_filename, 'w', encoding='utf-8') as ff:
            ff.write('\n\n'.join(conlls))


def count_data():
    with io.open(training_dest_filename, encoding='utf-8') as f:
        lines = f.read().split('\n\n')
        print len(lines) / 50


if __name__ == '__main__':
    init_dir()
    generate_dict_and_vectors()
    cws2conll(training_src_filename, training_dest_filename)
    cws2conll(test_src_filename, test_dest_filename)
    # count_data()
