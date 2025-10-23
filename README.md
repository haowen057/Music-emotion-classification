# DT2470 HT25 Music Informatics
# Group 10 : Hao Wen, Runze Cui, Shangxuan Tang, Weicheng Yuan

# Music Emotion Recognition - File Structure            
# 🎵 Music Emotion Recognition System

A deep learning-based system for real-time music emotion classification that analyzes 10-second audio segments.

## 🏗️ System Architecture

### Data Preprocessing (`EmotionSoundDataset`)
- Audio augmentation (random/deterministic modes)
- Automatic resampling to 22.05kHz
- 64-dimensional Log-Mel spectrogram feature extraction
- Support for 10 emotion label classifications

### Model Architecture (`CNNNetwork10`)
- **Network Structure**: Deep Convolutional Neural Network
- **Convolution Blocks**: 64→128→256→512 channels
- **Classifier**: 3 fully connected layers (1024→512→10)
- **Optimization**: Kaiming initialization + Progressive Dropout + Mixed Precision Training

### Data Augmentation Strategy
- **Audio Augmentation**: Gaussian noise, speed perturbation, pitch shifting
- **Spectrogram Augmentation**: Frequency masking, time masking
- **Dual Mode**: Random augmentation (training) + Deterministic augmentation (pre-computation)

### Training Pipeline
- **Optimizer**: AdamW + OneCycleLR scheduling
- **Data Flow**: Raw audio → Pre-computed augmentation → Stratified sampling → Real-time spectrogram augmentation → Model training

### Real-time Inference (`TenSecondEmotionClassifier`)
- 10-second real-time music emotion analysis
- PyAudio stream capture
- Multi-threaded processing architecture
- Support for real-time listening and file analysis

## ✨ Key Features

- 🚀 **Efficient Training**: Pre-computed features + Mixed precision acceleration
- 🛡️ **Enhanced Robustness**: Multi-dimensional data augmentation strategies
- ⚡ **Smart Optimization**: Adaptive learning rate + Early stopping mechanism
- 🎧 **Real-time Inference**: Low-latency 10-second audio analysis
- 🔧 **Deployment Friendly**: Comprehensive troubleshooting guide

## 📊 Dataset

- **Source**: MTG-Jamendo mood/theme dataset
- **Labels**: 10 music emotions (melodic, energetic, dark, film, relaxing, dream, ambiental, love, soundscape, emotional)
- **Scale**: Thousands of balanced samples
- **Format**: MP3 audio + TSV metadata

## 🚀 Quick Start

```python
# Real-time emotion detection
classifier = TenSecondEmotionClassifier("model.pth")
classifier.start_listening()




# 1. 📊 Data Processing & Exploration
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
