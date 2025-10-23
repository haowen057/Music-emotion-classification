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
        """
        10秒音乐情绪分类器
        
        参数:
            model_path: 训练好的模型路径
            sample_rate: 采样率 (与训练时一致)
        """
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.sample_rate = sample_rate
        self.chunk_size = sample_rate * 10  # 10秒音频
        
        # 加载模型
        self.model = self.load_model(model_path)
        self.model.eval()
        
        # 情绪标签 (与训练时一致)
        self.emotion_labels = Config.TOP10_TAGS
        
        # 音频缓冲区 - 存储10秒数据
        self.audio_buffer = deque(maxlen=self.chunk_size)
        self.is_listening = False
        self.audio_interface = None
        self.stream = None
        
        # 分析状态
        self.is_analyzing = False
        self.last_analysis_time = 0
        self.analysis_interval = 10  # 每10秒分析一次
        
        print(f"🎵 10秒音乐情绪分类器初始化完成")
        print(f"📊 设备: {self.device}")
        print(f"⏱️  分析时长: 10秒")
        print(f"🎯 支持的情绪类别: {self.emotion_labels}")
    
    def load_model(self, model_path):
        """加载训练好的模型"""
        model = CNNNetwork10(num_classes=len(Config.TOP10_TAGS)).to(self.device)
        
        try:
            checkpoint = torch.load(model_path, map_location=self.device)
            
            # 自动检测模型格式
            if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                state_dict = checkpoint['model_state_dict']
            else:
                state_dict = checkpoint
            
            model.load_state_dict(state_dict)
            print(f"✅ 模型加载成功: {model_path}")
            
        except Exception as e:
            print(f"❌ 模型加载失败: {e}")
            raise
        
        return model
    
    def preprocess_audio(self, audio_chunk):
        """
        音频预处理（与训练时保持一致）
        """
        # 转换为张量
        if isinstance(audio_chunk, np.ndarray):
            audio_tensor = torch.from_numpy(audio_chunk).float()
        else:
            audio_tensor = audio_chunk.clone().detach().float()
        
        # 确保形状为 [1, samples]
        if audio_tensor.dim() == 1:
            audio_tensor = audio_tensor.unsqueeze(0)
        
        # 裁剪或填充到10秒长度
        if audio_tensor.shape[1] > self.chunk_size:
            audio_tensor = audio_tensor[:, :self.chunk_size]
        elif audio_tensor.shape[1] < self.chunk_size:
            pad_size = self.chunk_size - audio_tensor.shape[1]
            audio_tensor = torch.nn.functional.pad(audio_tensor, (0, pad_size))
        
        return audio_tensor.to(self.device)
    
    def extract_logmel_features(self, audio_tensor):
        """
        提取Log-Mel特征（与训练时保持一致）
        """
        mel_transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=self.sample_rate,
            n_fft=1024,
            hop_length=512,
            n_mels=64
        ).to(self.device)
        
        mel_spectrogram = mel_transform(audio_tensor)
        logmel_features = torch.log(mel_spectrogram + 1e-9)
        
        return logmel_features.unsqueeze(0)  # 添加批次维度
    
    def classify_10s_emotion(self, audio_chunk):
        """
        分类10秒音频片段的情绪
        """
        with torch.no_grad():
            try:
                # 检查音频长度
                if len(audio_chunk) < self.chunk_size:
                    print(f"⚠️ 音频长度不足: {len(audio_chunk)}/{self.chunk_size} 采样点")
                    return "Insufficient Audio", 0.0
                
                # 预处理
                processed_audio = self.preprocess_audio(audio_chunk)
                
                # 提取特征
                features = self.extract_logmel_features(processed_audio)
                
                # 预测
                outputs = self.model(features)
                probabilities = torch.softmax(outputs, dim=1)
                confidence, predicted = torch.max(probabilities, 1)
                
                emotion_idx = predicted.item()
                confidence_score = confidence.item()
                
                return self.emotion_labels[emotion_idx], confidence_score
                
            except Exception as e:
                print(f"❌ 分类失败: {e}")
                return "Classification Error", 0.0
    
    def audio_callback(self, in_data, frame_count, time_info, status):
        """
        音频流回调函数
        """
        # 将音频数据添加到缓冲区
        audio_data = np.frombuffer(in_data, dtype=np.float32)
        self.audio_buffer.extend(audio_data)
        
        return (in_data, pyaudio.paContinue)
    
    def start_analysis_loop(self):
        """
        开始分析循环（在单独线程中运行）
        """
        def analysis_loop():
            buffer_fill_start = None
            
            while self.is_listening:
                current_buffer_size = len(self.audio_buffer)
                
                # 显示缓冲区填充进度
                if current_buffer_size < self.chunk_size:
                    if buffer_fill_start is None:
                        buffer_fill_start = time.time()
                        print(f"🔄 正在填充音频缓冲区... ({current_buffer_size}/{self.chunk_size})")
                    else:
                        progress = (current_buffer_size / self.chunk_size) * 100
                        if int(progress) % 10 == 0:  # 每10%显示一次
                            print(f"🔄 缓冲区填充: {progress:.0f}% ({current_buffer_size}/{self.chunk_size})")
                
                # 当缓冲区有10秒数据时进行分析
                if current_buffer_size >= self.chunk_size:
                    if buffer_fill_start:
                        fill_time = time.time() - buffer_fill_start
                        print(f"✅ 缓冲区填充完成! 耗时: {fill_time:.1f}秒")
                        buffer_fill_start = None
                    
                    # 获取10秒音频数据
                    chunk = np.array(list(self.audio_buffer))[-self.chunk_size:]
                    
                    print("🔍 正在分析10秒音频...")
                    start_time = time.time()
                    
                    # 分类情绪
                    emotion, confidence = self.classify_10s_emotion(chunk)
                    
                    analysis_time = time.time() - start_time
                    
                    # 显示结果
                    print("\n" + "="*60)
                    print(f"🎵 10秒音乐情绪分析结果:")
                    print(f"📊 检测情绪: {emotion}")
                    print(f"🎯 置信度: {confidence:.3f}")
                    print(f"⏱️  分析耗时: {analysis_time:.2f}秒")
                    print(f"🕒 时间: {time.strftime('%H:%M:%S')}")
                    print("="*60 + "\n")
                    
                    # 清空缓冲区，重新开始收集
                    self.audio_buffer.clear()
                    buffer_fill_start = time.time()
                
                time.sleep(0.1)  # 减少CPU占用
        
        # 启动分析线程
        self.analysis_thread = threading.Thread(target=analysis_loop)
        self.analysis_thread.daemon = True
        self.analysis_thread.start()
    
    def start_listening(self):
        """
        开始从扬声器实时监听
        """
        try:
            self.audio_interface = pyaudio.PyAudio()
            
            # 打开音频流（从扬声器/立体声混音输入）
            self.stream = self.audio_interface.open(
                format=pyaudio.paFloat32,
                channels=1,  # 单声道
                rate=self.sample_rate,
                input=True,
                frames_per_buffer=1024,
                stream_callback=self.audio_callback,
                input_device_index=None  # 使用默认输入设备
            )
            
            self.is_listening = True
            self.stream.start_stream()
            
            print("🔊 开始10秒音乐情绪检测...")
            print("💡 请播放音乐，系统将每10秒分析一次")
            print("⏹️  按 Ctrl+C 停止检测")
            print("-" * 50)
            
            # 启动分析循环
            self.start_analysis_loop()
            
        except Exception as e:
            print(f"❌ 音频设备初始化失败: {e}")
            self.print_troubleshooting_guide()
    
    def print_troubleshooting_guide(self):
        """打印故障排除指南"""
        print("\n🔧 故障排除指南:")
        print("1. Windows: 在声音设置中启用'立体声混音'")
        print("2. Mac: 安装Soundflower或BlackHole进行音频路由")
        print("3. 检查麦克风/录音权限")
        print("4. 确保音频驱动正常")
        print("5. 尝试不同的输入设备索引")
    
    def stop_listening(self):
        """停止监听"""
        self.is_listening = False
        if self.stream:
            self.stream.stop_stream()
            self.stream.close()
        if self.audio_interface:
            self.audio_interface.terminate()
        print("🛑 音乐情绪检测已停止")

# ==================== 使用示例 ====================
def main():
    # 模型路径
    MODEL_PATH = "best_emotion_classifier_amp_enhanced_11.pth"
    
    # 创建分类器
    classifier = TenSecondEmotionClassifier(model_path=MODEL_PATH)
    
    try:
        # 开始监听和分析
        classifier.start_listening()
        
        # 保持主线程运行
        while classifier.is_listening:
            time.sleep(1)
                
    except KeyboardInterrupt:
        print("\n⏹️ 用户中断检测")
    except Exception as e:
        print(f"❌ 运行错误: {e}")
    finally:
        classifier.stop_listening()

def test_with_pre_recorded_audio(audio_file_path):
    """
    使用预录音频文件测试分类器
    """
    classifier = TenSecondEmotionClassifier("best_emotion_classifier_amp_enhanced_11.pth")
    
    try:
        # 加载音频文件
        audio, sr = torchaudio.load(audio_file_path)
        if sr != classifier.sample_rate:
            resampler = torchaudio.transforms.Resample(sr, classifier.sample_rate)
            audio = resampler(audio)
        
        # 转换为单声道
        if audio.shape[0] > 1:
            audio = torch.mean(audio, dim=0, keepdim=True)
        
        # 确保音频长度足够
        if audio.shape[1] < classifier.chunk_size:
            print(f"⚠️ 音频文件过短: {audio.shape[1]/classifier.sample_rate:.1f}秒")
            return
        
        # 分析前10秒
        chunk = audio[:, :classifier.chunk_size]
        
        print("🔍 分析音频文件...")
        emotion, confidence = classifier.classify_10s_emotion(chunk.numpy())
        
        print("\n" + "="*50)
        print(f"🎵 音频文件分析结果:")
        print(f"📊 检测情绪: {emotion}")
        print(f"🎯 置信度: {confidence:.3f}")
        print("="*50)
            
    except Exception as e:
        print(f"❌ 文件测试失败: {e}")

if __name__ == "__main__":
    # 运行实时10秒检测
    main()
    
    # 取消注释以测试音频文件
    # test_with_pre_recorded_audio("your_audio_file.wav")