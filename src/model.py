# -*- coding: utf-8 -*-
import pickle
import io
from LSTM import LSTM_LayerSetting
from Activation import *
from CWS import CWS
from Data import Data

dataSet = 'msr'
path_lookup_table = '../PreTrainedWordEmbedding/charactor_OOVthr_50_%dv.txt' % 100
path_train_data = '../sighan2005/processed_wo_idioms/pro/%s_train.utf8' % dataSet
path_test_data = '../sighan2005/processed_wo_idioms/pro/%s_test.utf8' % dataSet
path_dev_data = None
dic_label = {'B': 1, 'E': 2, 'M': 3, 'S': 0}
data = Data(path_lookup_table=path_lookup_table,
            wordVecLen=100, path_train_data=path_train_data,
            path_test_data=path_test_data, path_dev_data=path_dev_data,
            flag_random_lookup_table=False,
            dic_label=dic_label,
            use_bigram_feature=False,
            random_seed=1234, flag_toy_data=False)


class Model(CWS):
    def __init__(self,
                 alpha=0.2,
                 squared_filter_length_limit=False,
                 batch_size=20,
                 n_epochs=60,
                 seg_result_file='seg_result/seg_result_msr',
                 L2_reg=0.0001,
                 HINGE_reg=0.2,
                 wordVecLen=100,
                 preWindowSize=1,
                 surWindowSize=2,
                 flag_dropout=(True,),
                 flag_dropout_scaleWeight=False,
                 layer_sizes=(100 * 3, 150),
                 dropout_rates=(0.2,),
                 layer_types=('LSTM',),
                 layer_setting=(LSTM_LayerSetting(gate_activation=Sigmoid(), cell_activation=Tanh()),),
                 data=data,
                 use_bias=True,
                 use_bigram_feature=False,
                 random_seed=1234,
                 model_path='../models/model_1.pkl'):
        CWS.__init__(self,
                     alpha,
                     squared_filter_length_limit,
                     batch_size,
                     n_epochs,
                     seg_result_file,
                     L2_reg,
                     HINGE_reg,
                     wordVecLen,
                     preWindowSize,
                     surWindowSize,
                     flag_dropout,
                     flag_dropout_scaleWeight,
                     layer_sizes,
                     dropout_rates,
                     layer_types,
                     layer_setting,
                     data,
                     use_bias,
                     use_bigram_feature,
                     random_seed)
        self.model_path = model_path
        self.__load_model()

    def __load_model(self):
        with open(self.model_path, mode='rb') as file:
            model = pickle.load(file)
        self.params = model['param']
        self.layers = model['layers']

    def inference(self, text):
        char_ids = [self.data.dic_c2idx[char] for char in text]
        result = self.decode_fun(char_ids, None, 1.0)
        labels = self.get_label(result)
        tokens = []
        token = ''
        for char, label in zip(text, labels):
            if label == 'S':
                if token:
                    raise Exception('B token')
                    # tokens.append(token)
                    # token = ''
                tokens.append(char)
            elif label == 'B':
                if token:
                    raise Exception('B token')
                    # tokens.append(token)
                token = char
            elif label == 'I':
                token += char
            else:
                token += char
                tokens.append(token)
                token = ''
        return tokens

    def get_label(self, result):
        sen_len = len(result)
        ret = [-1 for i in xrange(sen_len)]
        Max = None
        pos = 0
        cur = sen_len - 1
        for i in xrange(4):
            if result[cur][i] is None:
                continue
            if result[cur][i] > Max or Max is None:
                Max = result[cur][i]
                pos = i
        ret[sen_len - 1] = pos
        while cur != 0:
            pos = result[cur][pos + 4]
            cur -= 1
            ret[cur] = pos
        return [self.data.dic_idx2label[l] for l in ret]

    def evaluation(self, true_filename, pred_filename=None):
        true_positive_count = 0
        pred_count = 0
        true_count = 0

        if not pred_filename:
            with io.open(true_filename, encoding='utf-8') as f:
                lines = f.read().splitlines()
                for line in lines:
                    true_tokens = [token for token in line.split(u' ') if token]
                    pred_tokens = self.inference(line.replace(u' ', u''))
                    pred_count += len(pred_tokens)
                    true_count += len(true_tokens)
                    correct_spans = set(self.get_spans(true_tokens)).intersection(self.get_spans(pred_tokens))
                    true_positive_count += len(correct_spans)
        else:
            with io.open(true_filename, encoding='utf-8') as true_file:
                true_lines = true_file.read().splitlines()
                with io.open(pred_filename, encoding='utf-8') as pred_file:
                    pred_lines = pred_file.read().splitlines()
                    for pred_line, true_line in zip(pred_lines, true_lines):
                        true_tokens = [token for token in true_line.split(u' ') if token]
                        pred_tokens = [token for token in pred_line.split(u' ') if token]
                        pred_count += len(pred_tokens)
                        true_count += len(true_tokens)
                        correct_spans = set(self.get_spans(true_tokens)).intersection(self.get_spans(pred_tokens))
                        true_positive_count += len(correct_spans)

        print true_positive_count / pred_count, true_positive_count / true_count

    def get_spans(self, tokens):
        offset = 0
        spans = []
        for token in tokens:
            spans.append((offset, offset + len(token)))
            offset += len(token)
        return spans


if __name__ == '__main__':
    model = Model(model_path='../models/model_14.pkl')
    # ret = model.inference(u'我爱北京天安门。')
    # print '/'.join(ret)
    # model.evaluation('../data/msr_test.utf8')
    model.evaluation('../data/msr_test.utf8', './seg_result/seg_result_msr_14')
