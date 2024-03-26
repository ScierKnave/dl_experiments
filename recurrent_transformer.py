
import torch.nn as nn
import torch



class recurrent_transformer(nn.Module):
    def __init__(
            self,
            nb_layers, 
            vocab_size,
            batch_size,
            token_size,
            hidden_length,
            symbolic_length,
            gradient_horizon
    ):
        super(recurrent_transformer, self).__init__()

        # hypers
        self.gradient_horizon = gradient_horizon
        self.hidden_length = hidden_length
        self.batch_size = batch_size

        # transformer 
        self.embedding = nn.Embedding(vocab_size, token_size)
        decoder_layer = nn.TransformerDecoderLayer(d_model=token_size, nhead=4)
        self.transformer = nn.TransformerDecoder(decoder_layer, nb_layers)

        # contexts
        self.hidden_context = torch.rand(batch_size, hidden_length, token_size) # (B, S_H, T)
        self.hidden_counts = torch.zeros(batch_size, hidden_length) #(S)
        self.symbolic_context = torch.zeros(batch_size, symbolic_length, token_size) # (B, S_S, T)

        # dual output
        self.symbolic_out = nn.Linear(token_size*symbolic_length, vocab_size)
        self.hidden_out = nn.Linear(token_size*hidden_length, token_size)

    def autoregress(self, sequence, item, dim=1):
        #print(sequence.shape)
        #print(item.shape)

        return torch.cat((sequence[:, 1:], item), dim)

    def forward(self, symbols):

        symbols = self.embedding(symbols) # (B, 1) -> (B, T)

        self.autoregress(self.symbolic_context, symbols, -2) # -> (B, S_S)

        pre_out = self.transformer(self.symbolic_context, self.symbolic_context) # -> (B, S_S, T)
        s_out, h_out = (self.symbolic_out(pre_out.flatten(start_dim=1)), self.hidden_out(pre_out.flatten(start_dim=1)))

        self.autoregress(self.hidden_context, h_out.unsqueeze(1)) # add tought to sequence
        self.autoregress(self.hidden_counts, torch.zeros(self.batch_size, 1))

        # make gradient cuts past the gradient horizons
        for pos in range(self.hidden_length):
            if self.hidden_counts[0, pos] > self.gradient_horizon:
                hidden = self.hidden_context[:, pos, :]
                hidden = hidden.clone().detach()
                self.self.hidden_context[:, pos, :]


        # return probabilities for each vocab bit
        return torch.softmax(s_out, dim=-1)


