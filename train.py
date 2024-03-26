import torch
from transformers import GPT2Tokenizer
from recurrent_transformer import *


def train_sequence(model, optimizer, loss_function, train_tokens, sub_sq_length, batch_size, epochs, step_freq):
    model.train()

    for epoch in range(epochs):

        # batch the tokens
        B = batch_size
        S = train_tokens.shape[0]
        K = S // sub_sq_length

        train_tokens = train_tokens[:K * sub_sq_length] # cut the extra extra text
        train_tokens = torch.reshape(train_tokens, shape=[K, -1]) # (K, S)

        loss = 0

        for b in range(0, K//B):

            batch_tokens = train_tokens[b * B: (b + 1) * B] # (B, S)

            for i in range(sub_sq_length-1):
                token = torch.tensor(batch_tokens[:, i], dtype=torch.long).unsqueeze(1) # (B, 1)
                next_token = torch.tensor(batch_tokens[:, i + 1], dtype=torch.long) # (B)
                next_token_hat = model(token) # (B, vocab_size)
                #print('next_token_hat', next_token_hat.shape) # ()
                loss += loss_function(next_token_hat, next_token)

                if i % step_freq == 0:
                    print('loss:', loss)
                    loss.backward()
                    optimizer.step()
                    optimizer.zero_grad()
                    loss = 0
                    print("Training on batch...")
        

# getting data for training 
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
with open('data/shakespear.txt', 'r', encoding='utf-8') as file:
    shakespear_corpus = file.read()
tokens = torch.Tensor(tokenizer.encode(shakespear_corpus))


# training 
model = recurrent_transformer(
    nb_layers=2,
    batch_size=32,
    vocab_size=50257,
    token_size=512,
    symbolic_length=100,
    hidden_length=100,
    gradient_horizon=7
)

optimizer = torch.optim.AdamW(model.parameters())
loss_function = torch.nn.CrossEntropyLoss()


train_sequence(model, optimizer, loss_function, tokens, epochs=1, sub_sq_length=400, batch_size=32, step_freq=100)


    