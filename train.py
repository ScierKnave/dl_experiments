import torch
from transformers import GPT2Tokenizer
from recurrent_transformer import *


def train_sequence(model, optimizer, loss_function, train_tokens, epochs, step_freq):
    model.train()
    for epoch in range(epochs):

        loss = 0

        for i in range(0, len(train_tokens) - 1):
            token = torch.tensor(train_tokens[i], dtype=torch.long).unsqueeze(0)
            next_token = torch.tensor(train_tokens[i + 1], dtype=torch.long).unsqueeze(0)
            next_token_hat = model(token)
            loss += loss_function(next_token_hat, next_token)

            if i % step_freq == 0:
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
                loss = 0
        

# getting data for training 
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
with open('data/shakespear.txt', 'r', encoding='utf-8') as file:
    shakespear_corpus = file.read()
tokens = tokenizer.encode(shakespear_corpus)
print(tokens[:10])


# training 
model = recurrent_transformer(
    nb_layers=2,
    vocab_size=50257,
    token_size=512,
    symbolic_length=10,
    hidden_length=10,
    gradient_horizon=1
)

optimizer = torch.optim.AdamW(model.parameters())
loss_function = torch.nn.CrossEntropyLoss()


train_sequence(model, optimizer, loss_function, tokens, epochs=1, step_freq=100)


    