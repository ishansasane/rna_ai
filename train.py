import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from dataset import RNADataset
from model import RNAPredictor
import argparse

def train(data_dir, epochs=5, batch_size=16, lr=0.001, max_len=200):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    dataset = RNADataset(data_dir, max_len=max_len)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    model = RNAPredictor().to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    # We add a pos_weight of 100 because 99% of the map is empty. 
    # This forces the AI to pay 100x more attention to actual bonds!
    pos_weight = torch.tensor([100.0]).to(device)
    criterion = nn.BCEWithLogitsLoss(reduction='none', pos_weight=pos_weight)
    
    print(f"Starting training on dataset of size {len(dataset)}...")
    
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        valid_batches = 0
        
        for batch_idx, (seqs, contacts, masks, lengths) in enumerate(dataloader):
            seqs, contacts, masks = seqs.to(device), contacts.to(device), masks.to(device)
            
            if lengths.sum() == 0: continue
            
            optimizer.zero_grad()
            logits = model(seqs)
            
            loss = criterion(logits, contacts)
            # mask padded regions out of loss
            loss = (loss * masks).sum() / (masks.sum() + 1e-8)
            
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            valid_batches += 1
            
            if batch_idx % 10 == 0:
                print(f"Epoch {epoch+1}/{epochs} | Batch {batch_idx} | Loss: {loss.item():.4f}")
                
        avg_loss = total_loss / max(1, valid_batches)
        print(f"Epoch {epoch+1} completed. Average Loss: {avg_loss:.4f}")
        
    torch.save(model.state_dict(), "rna_model.pth")
    print("Model saved to rna_model.pth")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, required=True, help="Path to CT files directory")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=16)
    args = parser.parse_args()
    
    train(args.data_dir, args.epochs, args.batch_size)
