
import torch.nn as nn
import torch



class recurrent_transfomer(nn.Module):
    def __init__(
            self,
            nb_layers, 
            vocab_size,
            token_size,
            hidden_length,
            symbolic_length
    ):
        super(recurrent_transfomer, self).__init__()

        # transformer 
        decoder_layer = nn.TransformerDecoderLayer(d_model=token_size, nhead=4)
        self.transformer = nn.TransformerDecoder(decoder_layer, nb_layers)

        # contexts
        self.hidden_context = torch.rand(hidden_length, token_size) #
        self.hidden_counts = torch.zeros(hidden_length)
        self.symbolic_context = torch.zeros(symbolic_length, token_size)

        # dual output
        self.symbolic_out = nn.Linear(token_size*symbolic_length, vocab_size)
        self.hidden_out = nn.Linear(token_size*hidden_length, token_size)

    def autoregress(self, sequence, item, dim):
        sequence = sequence[:, 1:, :]
        sequence = torch.cat(sequence, item, dim=dim)
        return sequence

    def forward(self, symbol):

        self.autoregress(self.symbolic_context, symbol, 1)

        pre_out = self.transformer(self.symbolic_context)
        s_out, h_out = (self.symbolic_out(pre_out), self.hidden_out(pre_out))

        self.autoregress(self.hidden_context, h_out, 1) # add tought to sequence
        self.autoregress(self.hidden_counts, torch.tensor([0]), 0)

        # make gradient cuts past the gradient horizons
        for pos in range(self.hidden_counts.shape[0]):
            if self.hidden_counts[pos] > self.gradient_horizon:
                hidden = self.hidden_context[:, pos, :]
                hidden = hidden.clone().detach()
                self.self.hidden_context[:, pos, :]


        # return probabilities for each vocab bit
        return torch.softmax(s_out, dim=-1)


