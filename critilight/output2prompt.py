import torch

def pred2int(pred_tensor, capacity=20.0):
    real_float = pred_tensor * capacity
    real_int = torch.round(torch.clamp(real_float, min=0)).int()
    return real_int

def pred2str(queue_length):
    prompt = ""
    return prompt
