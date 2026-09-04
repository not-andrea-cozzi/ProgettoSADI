import torch
from collections import Counter

for split in ['train', 'val', 'test']:
    data = torch.load(f'Dataset/Train/merged_{split}.pt')
    mates = []
    for item in data:
        if hasattr(item, 'mate_n'):
            mate = int(item.mate_n) if torch.is_tensor(item.mate_n) else item.mate_n
            mates.append(mate)
    print(f"{split}: {len(data)} elementi, distribuzione: {Counter(mates)}")