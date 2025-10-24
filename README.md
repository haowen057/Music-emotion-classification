# DT2470 HT25 Music Informatics
# Group 10: Runze Cui, Hao Wen, Shangxuan Tang, Weicheng Yuan

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

## 📊 Dataset

- **Source**: MTG-Jamendo mood/theme dataset
- **Labels**: 10 music emotions (melodic, energetic, dark, film, relaxing, dream, ambiental, love, soundscape, emotional)
- **Scale**: Thousands of balanced samples
- **Format**: MP3 audio + TSV metadata

## 🗂️ File Structure & Implementation

### 0. 💡 Baseline Model
| File | Purpose | Features |
|------|---------|----------|
| `baseline_MLP.py` | Baseline MLP model | Traditional feature-based approach |

**Core Audio Features Extracted:**
- **Zero Crossing Rate**: Mean & STD
- **RMS Energy**: Mean & STD  
- **Spectral Centroid**: Mean & STD
- **Spectral Bandwidth**: Mean & STD
- **Spectral Contrast**: Mean & STD
- **Spectral Flatness**: Mean & STD
- **Spectral Rolloff**: Mean & STD
- **Statistical Features**: Skewness & Kurtosis
- **MFCC Features**: 13 coefficients + Delta coefficients

**Technical Implementation:**
```python
# Audio preprocessing
- Sample rate: Native
- Duration: 10 seconds (padded/truncated)
- Data type: float32
- NaN handling: Zero replacement

# Feature engineering
- 13 MFCC coefficients + 13 Delta MFCC
- 8 spectral features (each with mean/std)
- 2 statistical features
- Total: 42-dimensional feature vector
