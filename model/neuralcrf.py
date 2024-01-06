# 
# @author: Allan
#

import torch
import torch.nn as nn

from model.bilstm_encoder import BiLSTMEncoder
from model.linear_crf_inferencer import LinearCRF
from typing import Tuple
from overrides import overrides


class NNCRF(nn.Module):

    def __init__(self, config, print_info: bool = True):
        super(NNCRF, self).__init__()
        self.device = config.device
        self.encoder = BiLSTMEncoder(config, print_info=print_info)
        self.inferencer = LinearCRF(config)

    @overrides
    def forward(self, words: torch.Tensor, # shape(10,47)
                    word_seq_lens: torch.Tensor, # shape(10)
                    annotation_mask : torch.Tensor, # shape(10,47,20)
                    tags: torch.Tensor) -> torch.Tensor: # shape(10,47)
        """
        Calculate the negative loglikelihood.
        :param words: (batch_size x max_seq_len)
        :param word_seq_lens: (batch_size)
        :param chars: (batch_size x max_seq_len x max_char_len)
        :param char_seq_lens: (batch_size x max_seq_len)
        :param tags: (batch_size x max_seq_len)
        :return: the loss with shape (batch_size)
        """
        lstm_scores = self.encoder(words, word_seq_lens) # shape(10,47,20)
        batch_size = words.size(0)
        sent_len = words.size(1)
        maskTemp = torch.arange(1, sent_len + 1, dtype=torch.long).view(1, sent_len).expand(batch_size, sent_len).to(self.device)
        mask = torch.le(maskTemp, word_seq_lens.view(batch_size, 1).expand(batch_size, sent_len)).to(self.device)
        unlabed_score, labeled_score =  self.inferencer(lstm_scores, word_seq_lens, tags, mask)
        return unlabed_score - labeled_score

    def decode(self, batchInput: Tuple) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Decode the batch input
        :param batchInput:
        :return:
        """
        # shape(10,41)   shape(10)     shape(10,41,15)  shape(10,41)        None        shape(10,41)
        wordSeqTensor, wordSeqLengths, annotation_mask, tagSeqTensor = batchInput
        features = self.encoder(wordSeqTensor, wordSeqLengths) # shape(10,41,20)
        bestScores, decodeIdx = self.inferencer.decode(features, wordSeqLengths, annotation_mask)
        return bestScores, decodeIdx
