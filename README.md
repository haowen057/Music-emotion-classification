# DT2470 HT25 Music Informatics
# Group 10 : Hao Wen, Runze Cui, Shangxuan Tang, Weicheng Yuan

# Music Emotion Recognition - File Structure

## 1. 📊 Data Processing & Exploration
| File | Purpose | Features |
|------|---------|----------|
| `001test.ipynb` | Data analysis notebook | Audio segmentation, data cleaning, exploratory analysis |
| `Emotionsounddataset.py` | Dataset handler | Audio preprocessing, label mapping, data loading |

## 2. 🧠 Model Architecture
| File | Model | Architecture Details |
|------|-------|---------------------|
| `cnn.py` | CNNNetwork10 | Convolutional Neural Network for spectrogram analysis |

## 3. 🏋️ Training Pipeline
| File | Training Components | Augmentation Strategies |
|------|-------------------|------------------------|
| `AudioAugmentation.py` | Data augmentation | Audio time-stretching, pitch shifting, spectral masking |
| `Train_Final.py` | Main training script | Mixed precision, OneCycleLR, early stopping |

## 4. 🚀 Deployment & Evaluation
| File | Application | Functionality |
|------|-------------|---------------|
| `real_time_test.py` | Real-time system | Live audio emotion classification |
