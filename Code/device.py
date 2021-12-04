import torch


device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
print('Device: ', device)
