import torch.nn as nn
import torch
device = 'gpu' if torch.cuda.is_available() else 'cpu'

#
class recurrent_transformer(nn.Module):
    def __init__(
            self,
            nb_layers, 
            nb_heads,
            vocab_size,
            batch_size,
            token_size,
            hidden_length,
            symbolic_length,
            gradient_horizon
    ):
        super().__init__()

        # set attributes
        self.nb_layers = nb_layers
        self.nb_heads = nb_heads
        self.vocab_size = vocab_size
        self.batch_size = batch_size
        self.token_size = token_size
        self.hidden_length = hidden_length
        self.symbolic_length = symbolic_length
        self.gradient_horizon = gradient_horizon

        # transformer 
        self.embedding = nn.Embedding(vocab_size, token_size)
        decoder_layer = nn.TransformerDecoderLayer(d_model=token_size, nhead=nb_heads)
        self.transformer = nn.TransformerDecoder(decoder_layer, nb_layers)

        # contexts
        self.adjust_contexts(self.batch_size)

        # dual output
        self.symbolic_out = nn.Linear(token_size*symbolic_length, vocab_size)
        self.hidden_out = nn.Linear(token_size*symbolic_length, token_size)

    def autoregress(self, sequence, item):
        return torch.cat((sequence[:, item.shape[1]:], item), dim=1)
    
    def adjust_contexts(self, batch_size):
        # resets contexts with appropriate sizes
        assert(batch_size >= 1)
        self.hidden_context = torch.rand(batch_size, self.hidden_length, self.token_size) # (B, H, T)
        self.hidden_counts = torch.zeros(batch_size, self.hidden_length, 1) #(B, C, 1)
        self.symbolic_context = torch.zeros(batch_size, self.symbolic_length, self.token_size) # (B, S, T)
        self.batch_size = batch_size


    def prompt(self, input_text, tokenizer, output_limit=50):
        # prompt the model

        # prepare the model for evalution
        self.eval()
        if not self.batch_size == 1: self.adjust_contexts(1)

        # condition on initial text
        symbols = torch.tensor(tokenizer.encode(input_text)).unsqueeze(0) # -> (1, input_len)

        output_text = ""

        for _ in range(output_limit):
            symbols = torch.argmax(self.forward(symbols)) # (1, 1)
            output_text += tokenizer.decoder(symbols[0,0].item())

        return output_text
                


    def forward(self, symbols):

        # embed input
        symbols = self.embedding(symbols) # (B, L) -> (B, L, T)

        # auto regress on input symbols
        self.symbolic_context = self.autoregress(self.symbolic_context, symbols) # -> (B, S, T)

        # pass through network
        pre_out = self.transformer(self.symbolic_context, self.symbolic_context) # -> (B, S, T)
        pre_out = pre_out.flatten(start_dim=1) # (B, S, T) -> (B, ST)
        s_out = self.symbolic_out(pre_out) # (B, ST) -> (B, S)
        h_out = self.hidden_out(pre_out).unsqueeze(1) # (B, ST) -> (B, 1, T)

        # autoregress on thoughts
        self.hidden_context = self.autoregress(self.hidden_context, h_out) # add thought to sequence

        # make gradient cuts past the gradient horizons
        if self.training:
            fresh_counter = torch.zeros(self.batch_size, 1, 1).to(device)
            self.hidden_counts = self.autoregress(self.hidden_counts, item=fresh_counter)
            for pos in range(self.hidden_length):
                if self.hidden_counts[0, pos] > self.gradient_horizon:
                    hidden = self.hidden_context[:, pos, :] # (B, 1, T)
                    hidden = hidden.clone().detach() # (B, 1, T)
                    self.hidden_context[:, pos, :] = hidden

        # return logit probabilities for each word in vocab
        return s_out


