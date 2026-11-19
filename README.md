# LandslideGuard AI
### AI-Powered Landslide Risk Monitoring System

LandslideGuard AI is a deep learning-based application designed to identify potential landslide-affected regions from satellite imagery using a U-Net segmentation model.

## Features

- Satellite image visualization
- AI-based landslide segmentation
- Landslide mask and probability map visualization
- Prediction reports in PDF and ZIP formats
- Prediction history
- Model performance metrics
- Interactive Streamlit dashboard

## Technology Stack

- Python
- PyTorch
- Streamlit
- NumPy and Pandas
- Matplotlib
- Raster/HDF5 data processing

## Model Information

- Architecture: U-Net
- Input channels: 14
- Output channels: 1
- Input resolution: 128 × 128
- Best validation epoch: 7

## Installation

Clone the repository:

```bash
git clone https://github.com/VanshPathania981/LandslideGuard-AI.git
cd LandslideGuard-AI
```

Create and activate a virtual environment:

```bash
python -m venv venv
```

Windows:

```powershell
.\venv\Scripts\Activate.ps1
```

Install dependencies:

```bash
pip install -r requirements.txt
```

## Run the Application

```bash
streamlit run app.py
```

## Project Structure

```text
LandslideGuard-AI/
├── app.py
├── model.py
├── models/
│   └── best_unet.pth
├── .streamlit/
│   └── config.toml
├── requirements.txt
├── normalization.json
├── prediction_history.py
├── report_generator.py
├── test_model.py
└── README.md
└── TESTING_RESULTS.md
```

## Important Note

This project is intended for academic demonstration and research purposes. Predictions should not be treated as official landslide warnings or a substitute for expert geological assessment.

## Author

Developed as an individual academic project at Lovely Professional University.
