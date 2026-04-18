import os
import torch
from torch.utils.data import Dataset
import numpy as np

def parse_ct_file(filepath):
    with open(filepath, 'r') as f:
        lines = f.readlines()
        
    if not lines: return None, None
    try:
        length = int(lines[0].strip().split()[0])
    except:
        return None, None
        
    seq = []
    pairs = []
    for line in lines[1:]:
        parts = line.strip().split()
        if len(parts) >= 5:
            base = parts[1].upper()
            try:
                pair = int(parts[4]) - 1
            except:
                continue
            seq.append(base)
            pairs.append(pair)
            
    if len(seq) != length: return None, None
        
    contact_map = np.zeros((length, length), dtype=np.float32)
    for i, p in enumerate(pairs):
        if p != -2: # -1 indicates no pair, we subtracted 1 so it's -2
            if p >= 0 and p < length:
                contact_map[i, p] = 1.0
                contact_map[p, i] = 1.0
            
    return "".join(seq), contact_map

class RNADataset(Dataset):
    def __init__(self, data_dir, max_len=200):
        self.data_dir = data_dir
        self.files = [os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.endswith('.ct')]
        self.max_len = max_len
        self.char2idx = {'A': 0, 'C': 1, 'G': 2, 'U': 3, 'T': 3, 'N': 4}
        
    def __len__(self):
        return len(self.files)
        
    def __getitem__(self, idx):
        seq, contact_map = parse_ct_file(self.files[idx])
        
        if seq is None or len(seq) > self.max_len:
            # Return dummy if too long or invalid
            return torch.ones(self.max_len, dtype=torch.long)*4, torch.zeros((self.max_len, self.max_len), dtype=torch.float32), torch.zeros((self.max_len, self.max_len), dtype=torch.float32), 0
            
        encoded = [self.char2idx.get(c, 4) for c in seq]
        L = len(encoded)
        
        padded_encoded = np.ones(self.max_len, dtype=np.int64) * 4
        padded_encoded[:L] = encoded
        
        padded_contact = np.zeros((self.max_len, self.max_len), dtype=np.float32)
        padded_contact[:L, :L] = contact_map
        
        mask = np.zeros((self.max_len, self.max_len), dtype=np.float32)
        mask[:L, :L] = 1.0
        
        return torch.tensor(padded_encoded), torch.tensor(padded_contact), torch.tensor(mask), L
