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
            super(recurrent_transformer, self).__init__()

            # hypers
            self.gradient_horizon = gradient_horizon
            self.hidden_length = hidden_length
            self.batch_size = batch_size

            # transformer 
            self.embedding = nn.Embedding(vocab_size, token_size)
            decoder_layer = nn.TransformerDecoderLayer(d_model=token_size, nhead=nb_heads)
            self.transformer = nn.TransformerDecoder(decoder_layer, nb_layers)

            # contexts
            self.adjust_contexts()

            # dual output
            self.symbolic_out = nn.Linear(token_size*symbolic_length, vocab_size)
            self.hidden_out = nn.Linear(token_size*symbolic_length, token_size)

        def autoregress(self, sequence, item,):
            return torch.cat((sequence[:, 1:, :], item))
        
        def adjust_contexts(self, batch_size):
            # resets contexts with appropriate sizes
            self.hidden_context = torch.rand(batch_size, self.hidden_length, self.token_size) # (B, H, T)
            self.hidden_counts = torch.zeros(batch_size, self.hidden_length, 1) #(B, C, 1)
            self.symbolic_context = torch.zeros(batch_size, self.symbolic_length, self.token_size) # (B, S, T)
            self.batch_size = batch_size


        def prompt(self, text, limit=50):
            
            # prompt the model

            self.eval()
            if not self.batch_size == 1: self.adjust_contexts(1)
            with torch.no_grad:
                # TODO: complete
                for token_gen in range(limit):
                    # adjust with respect to input
                    if symbols.dim() == 1: symbols = symbols.unsqueeze(0)
                    symbols = self.embedding(symbols) # (B, S, 1) -> (B, S, T)
                    B, S, T = symbols.shape

                    self.autoregress(self.symbolic_context, symbols) # -> (B, S, T)

                    pre_out = self.transformer(self.symbolic_context, self.symbolic_context) # -> (B, S, T)
                    pre_out = pre_out.flatten(start_dim=1) # (B, S, T) -> (BS, T)
                    s_out = self.symbolic_out(pre_out) 
                    h_out = self.hidden_out(pre_out)

                    self.autoregress(self.hidden_context, h_out.unsqueeze(1)) # add thought to sequence


        def forward(self, symbols):
            
            # adjust with respect to input
            if symbols.dim() == 1: symbols = symbols.unsqueeze(0)
            symbols = self.embedding(symbols) # (B, S, 1) -> (B, S, T)
            B, S, T = symbols.shape

            self.autoregress(self.symbolic_context, symbols) # -> (B, S, T)

            pre_out = self.transformer(self.symbolic_context, self.symbolic_context) # -> (B, S, T)
            pre_out = pre_out.flatten(start_dim=1) # (B, S, T) -> (BS, T)
            s_out = self.symbolic_out(pre_out) 
            h_out = self.hidden_out(pre_out)

            self.autoregress(self.hidden_context, h_out.unsqueeze(1)) # add thought to sequence
            self.autoregress(self.hidden_counts, torch.zeros(self.batch_size, 1).to(device))

            # make gradient cuts past the gradient horizons
            for pos in range(self.hidden_length):
                if self.hidden_counts[0, pos] > self.gradient_horizon:
                    hidden = self.hidden_context[:, pos, :] # (B, 1, T)
                    hidden = hidden.clone().detach() # (B, 1, T)
                    self.hidden_context[:, pos, :] = hidden


            # return logit probabilities for each vocab bit
            return s_out


