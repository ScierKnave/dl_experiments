import torch

model_path = ""

model = torch.load(model_path)

def promt_model(model):
    while True:
        user_text = input("User:")
        print("Circuit:", model.prompt(user_text))
