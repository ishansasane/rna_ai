#!/bin/bash
echo "Setting up RNA Predictor project..."
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
echo "Setup complete! You can now run the web app with:"
echo "source venv/bin/activate"
echo "python app.py"
echo ""
echo "To train the model with your dataset, run:"
echo "source venv/bin/activate"
echo "python train.py --data_dir /home/ishan/Documents/honors_project/RNA_STRAND_data/all_ct_files --epochs 1"
