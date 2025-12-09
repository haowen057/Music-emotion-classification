import torch
import torchaudio
import numpy as np
import pyaudio
import threading
import time
from collections import deque
from cnn import CNNNetwork10
from Train_Final import Config


class TenSecondEmotionClassifier:
    def __init__(self, model_path, sample_rate=22050):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.sample_rate = sample_rate
        self.chunk_size = sample_rate * 10
        
        self.model = self.load_model(model_path)
        self.model.eval()
        
        self.emotion_labels = Config.TOP10_TAGS
        
        self.audio_buffer = deque(maxlen=self.chunk_size)
        self.is_listening = False
        self.audio_interface = None
        self.stream = None
        
        self.is_analyzing = False
        self.last_analysis_time = 0
        self.analysis_interval = 10
        
        print(f"10-second music emotion classifier initialized")
        print(f"Device: {self.device}")
        print(f"Analysis duration: 10 seconds")
        print(f"Supported emotion categories: {self.emotion_labels}")
    
    def load_model(self, model_path):
        model = CNNNetwork10(num_classes=len(Config.TOP10_TAGS)).to(self.device)
        
        try:
            checkpoint = torch.load(model_path, map_location=self.device)
            
            if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                state_dict = checkpoint['model_state_dict']
            else:
                state_dict = checkpoint
            
            model.load_state_dict(state_dict)
            print(f"Model loaded successfully: {model_path}")
            
        except Exception as e:
            print(f"Model loading failed: {e}")
            raise
        
        return model
    
    def preprocess_audio(self, audio_chunk):
        if isinstance(audio_chunk, np.ndarray):
            audio_tensor = torch.from_numpy(audio_chunk).float()
        else:
            audio_tensor = audio_chunk.clone().detach().float()
        
        if audio_tensor.dim() == 1:
            audio_tensor = audio_tensor.unsqueeze(0)
        
        if audio_tensor.shape[1] > self.chunk_size:
            audio_tensor = audio_tensor[:, :self.chunk_size]
        elif audio_tensor.shape[1] < self.chunk_size:
            pad_size = self.chunk_size - audio_tensor.shape[1]
            audio_tensor = torch.nn.functional.pad(audio_tensor, (0, pad_size))
        
        return audio_tensor.to(self.device)
    
    def extract_logmel_features(self, audio_tensor):
        mel_transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=self.sample_rate,
            n_fft=1024,
            hop_length=512,
            n_mels=64
        ).to(self.device)
        
        mel_spectrogram = mel_transform(audio_tensor)
        logmel_features = torch.log(mel_spectrogram + 1e-9)
        
        return logmel_features.unsqueeze(0)
    
    def classify_10s_emotion(self, audio_chunk):
        with torch.no_grad():
            try:
                if len(audio_chunk) < self.chunk_size:
                    print(f"Audio length insufficient: {len(audio_chunk)}/{self.chunk_size} samples")
                    return "Insufficient Audio", 0.0
                
                processed_audio = self.preprocess_audio(audio_chunk)
                
                features = self.extract_logmel_features(processed_audio)
                
                outputs = self.model(features)
                probabilities = torch.softmax(outputs, dim=1)
                confidence, predicted = torch.max(probabilities, 1)
                
                emotion_idx = predicted.item()
                confidence_score = confidence.item()
                
                return self.emotion_labels[emotion_idx], confidence_score
                
            except Exception as e:
                print(f"Classification failed: {e}")
                return "Classification Error", 0.0
    
    def audio_callback(self, in_data, frame_count, time_info, status):
        audio_data = np.frombuffer(in_data, dtype=np.float32)
        self.audio_buffer.extend(audio_data)
        
        return (in_data, pyaudio.paContinue)
    
    def start_analysis_loop(self):
        def analysis_loop():
            buffer_fill_start = None
            
            while self.is_listening:
                current_buffer_size = len(self.audio_buffer)
                
                if current_buffer_size < self.chunk_size:
                    if buffer_fill_start is None:
                        buffer_fill_start = time.time()
                        print(f"Filling audio buffer... ({current_buffer_size}/{self.chunk_size})")
                    else:
                        progress = (current_buffer_size / self.chunk_size) * 100
                        if int(progress) % 10 == 0:
                            print(f"Buffer fill: {progress:.0f}% ({current_buffer_size}/{self.chunk_size})")
                
                if current_buffer_size >= self.chunk_size:
                    if buffer_fill_start:
                        fill_time = time.time() - buffer_fill_start
                        print(f"Buffer fill completed! Time taken: {fill_time:.1f} seconds")
                        buffer_fill_start = None
                    
                    chunk = np.array(list(self.audio_buffer))[-self.chunk_size:]
                    
                    print("Analyzing 10-second audio...")
                    start_time = time.time()
                    
                    emotion, confidence = self.classify_10s_emotion(chunk)
                    
                    analysis_time = time.time() - start_time
                    
                    print("\n" + "="*60)
                    print(f"10-second Music Emotion Analysis Result:")
                    print(f"Detected Emotion: {emotion}")
                    print(f"Confidence: {confidence:.3f}")
                    print(f"Analysis Time: {analysis_time:.2f} seconds")
                    print(f"Time: {time.strftime('%H:%M:%S')}")
                    print("="*60 + "\n")
                    
                    self.audio_buffer.clear()
                    buffer_fill_start = time.time()
                
                time.sleep(0.1)
        
        self.analysis_thread = threading.Thread(target=analysis_loop)
        self.analysis_thread.daemon = True
        self.analysis_thread.start()
    
    def start_listening(self):
        try:
            self.audio_interface = pyaudio.PyAudio()
            
            self.stream = self.audio_interface.open(
                format=pyaudio.paFloat32,
                channels=1,
                rate=self.sample_rate,
                input=True,
                frames_per_buffer=1024,
                stream_callback=self.audio_callback,
                input_device_index=None
            )
            
            self.is_listening = True
            self.stream.start_stream()
            
            print("Starting 10-second music emotion detection...")
            print("Please play music, system will analyze every 10 seconds")
            print("Press Ctrl+C to stop detection")
            print("-" * 50)
            
            self.start_analysis_loop()
            
        except Exception as e:
            print(f"Audio device initialization failed: {e}")
            self.print_troubleshooting_guide()
    
    def print_troubleshooting_guide(self):
        print("\nTroubleshooting Guide:")
        print("1. Windows: Enable 'Stereo Mix' in sound settings")
        print("2. Mac: Install Soundflower or BlackHole for audio routing")
        print("3. Check microphone/recording permissions")
        print("4. Ensure audio drivers are working")
        print("5. Try different input device indexes")
    
    def stop_listening(self):
        self.is_listening = False
        if self.stream:
            self.stream.stop_stream()
            self.stream.close()
        if self.audio_interface:
            self.audio_interface.terminate()
        print("Music emotion detection stopped")

def main():
    MODEL_PATH = "best_emotion_classifier_amp_enhanced_11.pth"
    
    classifier = TenSecondEmotionClassifier(model_path=MODEL_PATH)
    
    try:
        classifier.start_listening()
        
        while classifier.is_listening:
            time.sleep(1)
                
    except KeyboardInterrupt:
        print("\nUser interrupted detection")
    except Exception as e:
        print(f"Runtime error: {e}")
    finally:
        classifier.stop_listening()

def test_with_pre_recorded_audio(audio_file_path):
    classifier = TenSecondEmotionClassifier("best_emotion_classifier_amp_enhanced_11.pth")
    
    try:
        audio, sr = torchaudio.load(audio_file_path)
        if sr != classifier.sample_rate:
            resampler = torchaudio.transforms.Resample(sr, classifier.sample_rate)
            audio = resampler(audio)
        
        if audio.shape[0] > 1:
            audio = torch.mean(audio, dim=0, keepdim=True)
        
        if audio.shape[1] < classifier.chunk_size:
            print(f"Audio file too short: {audio.shape[1]/classifier.sample_rate:.1f} seconds")
            return
        
        chunk = audio[:, :classifier.chunk_size]
        
        print("Analyzing audio file...")
        emotion, confidence = classifier.classify_10s_emotion(chunk.numpy())
        
        print("\n" + "="*50)
        print(f"Audio File Analysis Result:")
        print(f"Detected Emotion: {emotion}")
        print(f"Confidence: {confidence:.3f}")
        print("="*50)
            
    except Exception as e:
        print(f"File test failed: {e}")

if __name__ == "__main__":
    main()

