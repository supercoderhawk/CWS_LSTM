# -*- coding: utf-8 -*-
from LSTM import LSTM_HiddenLayer
from RNN import RNN_HiddenLayer
from GRU import GRU_HiddenLayer
from MLP import MLP_HiddenLayer


def Add_HiddenLayer(alpha,
                    squared_filter_length_limit,
                    L2_reg,
                    flag_dropout,
                    n_in, n_out, use_bias,
                    dropout_rate,
                    flag_dropout_scaleWeight,
                    layer_setting,
                    rng,
                    layer_type):
    if layer_type == 'RNN':
        return RNN_HiddenLayer(alpha,
                               squared_filter_length_limit,
                               L2_reg,
                               flag_dropout,
                               n_in, n_out, use_bias,
                               dropout_rate,
                               flag_dropout_scaleWeight,
                               layer_setting,
                               rng)
    elif layer_type == 'GRU':
        return GRU_HiddenLayer(alpha,
                               squared_filter_length_limit,
                               L2_reg,
                               flag_dropout,
                               n_in, n_out, use_bias,
                               dropout_rate,
                               flag_dropout_scaleWeight,
                               layer_setting,
                               rng)
    elif layer_type == 'LSTM':
        return LSTM_HiddenLayer(alpha,
                                squared_filter_length_limit,
                                L2_reg,
                                flag_dropout,
                                n_in, n_out, use_bias,
                                dropout_rate,
                                flag_dropout_scaleWeight,
                                layer_setting,
                                rng)
    elif layer_type == 'MLP':
        return MLP_HiddenLayer(alpha,
                               squared_filter_length_limit,
                               L2_reg,
                               flag_dropout,
                               n_in, n_out, use_bias,
                               dropout_rate,
                               flag_dropout_scaleWeight,
                               layer_setting,
                               rng)