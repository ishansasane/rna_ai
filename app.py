import torch
import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
import uvicorn
from model import RNAPredictor
import os

app = FastAPI()

app.mount("/static", StaticFiles(directory="static"), name="static")

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = RNAPredictor().to(device)

if os.path.exists("rna_model.pth"):
    model.load_state_dict(torch.load("rna_model.pth", map_location=device))
    print("Loaded trained model.")
else:
    print("No pre-trained model found. Using untrained weights for demonstration.")
model.eval()

char2idx = {'A': 0, 'C': 1, 'G': 2, 'U': 3, 'T': 3, 'N': 4}

class PredictRequest(BaseModel):
    sequence: str

@app.get("/")
def read_root():
    with open("static/index.html") as f:
        return HTMLResponse(content=f.read(), status_code=200)

@app.post("/predict")
def predict(req: PredictRequest):
    seq = req.sequence.upper().strip()
    if not seq:
        raise HTTPException(status_code=400, detail="Empty sequence")
    if len(seq) > 500:
        raise HTTPException(status_code=400, detail="Sequence too long (max 500)")
        
    encoded = [char2idx.get(c, 4) for c in seq]
    L = len(encoded)
    
    with torch.no_grad():
        x = torch.tensor([encoded], dtype=torch.long).to(device)
        logits = model(x)
        probs = torch.sigmoid(logits)[0].cpu().numpy()
        
    pairs = []
    for i in range(L):
        for j in range(i + 1, L):
            # Lowered threshold from 0.5 to 0.1 to account for high class imbalance
            if probs[i, j] > 0.1:
                pairs.append([i, j, float(probs[i, j])])
                
    return {
        "sequence": seq,
        "length": L,
        "pairs": pairs,
        "contact_map": probs.tolist()
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
