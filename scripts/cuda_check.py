import torch

cuda_available = torch.cuda.is_available()
print(f'cuda available in torch: {cuda_available}')

if cuda_available:
    print(f'num gpus: {torch.cuda.device_count()}')
