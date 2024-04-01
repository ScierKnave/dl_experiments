import torch
from transformers import GPT2Tokenizer
from recurrent_transformer import *
import yaml
import datetime

def train_sequence(model, 
                   optimizer, 
                   loss_function, 
                   train_tokens, 
                   sub_sq_length, 
                   batch_size, 
                   epochs, 
                   step_freq,
                   model_save_freq
                   ):
    
    model.train()

    grad_step_count = 0

    for epoch in range(epochs):

        # batch the tokens
        B = batch_size
        S = train_tokens.shape[0]
        K = S // sub_sq_length

        train_tokens = train_tokens[:K * sub_sq_length] # cut the extra extra text
        train_tokens = torch.reshape(train_tokens, shape=[K, -1]) # (K, S)

        loss = 0

        for b in range(0, K//B):
            
            # ensure gradients are cleared
            optimizer.zero_grad()   
            loss = 0

            batch_tokens = train_tokens[b * B: (b + 1) * B] # (B, S)

            for i in range(sub_sq_length-1):

                
                # get tokens
                tokens = torch.tensor(batch_tokens[:, i], dtype=torch.long).unsqueeze(1) # (B, 1)
                next_tokens = torch.tensor(batch_tokens[:, i + 1], dtype=torch.long) # (B)

                # forward pass
                next_tokens_hat = model(tokens) # (B, vocab_size) unormalized probs

                # add to error to loss
                loss += loss_function(next_tokens_hat, next_tokens)

                if (i+1) % step_freq == 0:
                    print('loss:', loss)
                    loss.backward()
                    optimizer.step()
                    grad_step_count+=1
                    loss = 0
                    optimizer.zero_grad()
                    model.zero_grad()
                    print("Training on batch...")

                if (grad_step_count+1) % model_save_freq == 0:
                    current_time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
                    save_path = f"model_{current_time}_step_{grad_step_count}.pt"
                    torch.save(model, save_path)
                    print(f"Saved model at step {grad_step_count} to {save_path}")

       


# import configuration
with open('reccurent_transformer/config_small.yaml', 'r') as file: 
    config = yaml.safe_load(file)

# getting data for training 
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
with open(**config['data']) as file: shakespear_corpus = file.read()
tokens = torch.Tensor(tokenizer.encode(shakespear_corpus))

# create model
model = recurrent_transformer(**config['model'])

# train the model
optimizer = torch.optim.AdamW(model.parameters())
loss_function = torch.nn.CrossEntropyLoss()
train_sequence(**config['training'], 
               model=model, 
               optimizer=optimizer, 
               loss_function=loss_function,
               train_tokens=tokens)


    