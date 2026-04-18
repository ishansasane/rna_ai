# RNA Structure Predictor

This is a complete full-stack deep learning project for RNA Secondary Structure Prediction, built for your honors project.

It features:
- **PyTorch Deep Learning Model**: A Bi-directional LSTM followed by a 2D Convolutional neural network that learns base-pairing probabilities from sequence embeddings.
- **Custom Dataset Parser**: Parses `.ct` (connectivity table) files provided in the `RNA_STRAND_data` directory.
- **FastAPI Web Backend**: An ultra-fast Python web server to serve the model predictions.
- **Modern Glassmorphism UI**: A stunning dark-mode web application using vanilla HTML/CSS/JS and Plotly.js to render predicted contact maps.

## Setup Instructions

1. Make sure Python 3 is installed.
2. Run the provided setup script to create a virtual environment and install dependencies:
```bash
chmod +x setup.sh
./setup.sh
```

## Running the Web App

Once installed, start the web application:
```bash
source venv/bin/activate
python app.py
```
Then, open your web browser and navigate to: `http://localhost:8000`

## Training the Model

To train the model on the `RNA_STRAND_data` dataset, run:
```bash
source venv/bin/activate
python train.py --data_dir ../RNA_STRAND_data/all_ct_files --epochs 5 --batch_size 16
```
The trained model will be saved as `rna_model.pth`. When you restart the web application (`python app.py`), it will automatically load this trained model.
