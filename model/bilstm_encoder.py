
import torch
import torch.nn as nn

from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from overrides import overrides

class BiLSTMEncoder(nn.Module):

    def __init__(self, config, print_info: bool = True):
        super(BiLSTMEncoder, self).__init__()

        self.label_size = config.label_size
        self.device = config.device

        self.label2idx = config.label2idx
        self.labels = config.idx2labels

        self.input_size = config.embedding_dim # 100

        self.word_embedding = nn.Embedding.from_pretrained(torch.FloatTensor(config.word_embedding), freeze=False).to(self.device) # shape(25305,100)
        self.word_drop = nn.Dropout(config.dropout).to(self.device)

        if print_info:
            print("[Model Info] Input size to LSTM: {}".format(self.input_size))
            print("[Model Info] LSTM Hidden Size: {}".format(config.hidden_dim))

        self.lstm = nn.LSTM(self.input_size, config.hidden_dim // 2, num_layers=1, batch_first=True, bidirectional=True).to(self.device)

        self.drop_lstm = nn.Dropout(config.dropout).to(self.device)

        final_hidden_dim = config.hidden_dim # 200

        if print_info:
            print("[Model Info] Final Hidden Size: {}".format(final_hidden_dim))

        self.hidden2tag = nn.Linear(final_hidden_dim, self.label_size).to(self.device)

    @overrides
    def forward(self, word_seq_tensor: torch.Tensor, # shape(10,47)
                       word_seq_lens: torch.Tensor) -> torch.Tensor: # shape(10)
        """
        Encoding the input with BiLSTM
        :param word_seq_tensor: (batch_size, sent_len)   NOTE: The word seq actually is already ordered before come here.
        :param word_seq_lens: (batch_size, 1)
        :return: emission scores (batch_size, sent_len, hidden_dim)
        """

        word_emb = self.word_embedding(word_seq_tensor) # shape(10,47,100)

        word_rep = self.word_drop(word_emb)


        sorted_seq_len, permIdx = word_seq_lens.sort(0, descending=True)
        _, recover_idx = permIdx.sort(0, descending=False)
        sorted_seq_tensor = word_rep[permIdx] # shape(10,47,150)

        packed_words = pack_padded_sequence(sorted_seq_tensor, sorted_seq_len.cpu(), True)
        lstm_out, _ = self.lstm(packed_words, None)
        lstm_out, _ = pad_packed_sequence(lstm_out, batch_first=True)  ## CARE: make sure here is batch_first, otherwise need to transpose.
        feature_out = self.drop_lstm(lstm_out) # shape(10,47,200)

        outputs = self.hidden2tag(feature_out) # shape(10,47,20)

        return outputs[recover_idx]


