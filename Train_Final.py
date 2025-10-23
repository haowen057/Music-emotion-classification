import os
import torch
import torchaudio
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
from tqdm import tqdm
import shutil
from cnn import CNNNetwork10
from AudioAugmentation import CombinedAugmentation
from Emotionsounddataset import EmotionSoundDataset
from sklearn.model_selection import train_test_split

# ==================== 配置参数 ====================
class Config:
    # 设备配置
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    
    # 训练参数
    BATCH_SIZE = 64
    EPOCHS = 120
    LEARNING_RATE = 1e-3
    WEIGHT_DECAY = 1e-4
    PATIENCE = 10
    
    # 音频参数
    SAMPLE_RATE = 22050
    NUM_SAMPLES = SAMPLE_RATE * 10  # 10秒音频
    
    # 学习率调度
    WARMUP_RATIO = 0.1
    MAX_LR_FACTOR = 1.5
    INIT_DIV_FACTOR = 25
    FINAL_DIV_FACTOR = 1e4
    
    # 路径配置
    BEST_MODEL_PATH = "best_emotion_classifier_amp_enhanced_11.pth"
    ANNOTATIONS_FILE = r"C:\Users\14217\Desktop\DT2470\Predictions with sound classifier\metadata_balanced_10s_multi.tsv"
    AUDIO_DIR = r"C:\Users\14217\Desktop\DT2470\Predictions with sound classifier"
    PRECOMPUTE_DIR = os.path.join(AUDIO_DIR, "precomputed")
    
    # 标签配置
    TOP10_TAGS = [
        'melodic', 
        'energetic', 
        'dark', 
        'film', 
        'relaxing',
        'dream', 
        'ambiental', 
        'love', 
        'soundscape', 
        'emotional'
    ]

# ==================== 预计算模块 ====================
class FeaturePrecomputer:
    """特征预计算器"""
    
    @staticmethod
    def precompute_features(dataset, save_dir, suffix=""):
        """预计算特征（增强完全由数据集类控制）"""
        features = []
        labels = []
        
        # 直接从数据集获取增强状态
        augment_info = dataset.get_dataset_info()
        print(f"🔧 预计算特征 | 音频增强: {augment_info['audio_augment']} | 频谱增强: {augment_info['spec_augment']}")
        
        for i in tqdm(range(len(dataset))):
            try:
                # 🎯 简化：直接使用标准接口，让数据集处理所有增强逻辑
                mel, label = dataset[i]
                
                # 确保正确的维度
                if mel.ndim == 2:
                    mel = mel.unsqueeze(0)
                elif mel.ndim == 3 and mel.shape[0] != 1:
                    mel = mel.mean(0, keepdim=True)
                
                # detach() 移除梯度信息, numpy()转换为可保存格式, cpu()最终转载位置
                # 保留预计算特征,后续方便调用
                features.append(mel.cpu().detach().numpy())  # 添加 .detach()
                labels.append(label)
                
            except Exception as e:
                print(f"⚠️ 跳过样本 {i}: {e}")
                continue
            
        # 保存特征
        features_array = np.stack(features).astype(np.float32)
        labels_array = np.array(labels)
    
        features_path = os.path.join(save_dir, f"features_{suffix}.npy")
        labels_path = os.path.join(save_dir, f"labels_{suffix}.npy")
        
        np.save(features_path, features_array)
        np.save(labels_path, labels_array)
        
        print(f"✅ 预计算完成: {len(features_array)} 样本")
        return features_path, labels_path
    

# ==================== 数据集类 ====================
class PrecomputedTrainDataset(Dataset):
    """
    预计算训练数据集（包含实时频谱增强）
    """
    
    def __init__(self, features, labels, device):  # 🎯 修复：直接传入数组
        self.features = features
        self.labels = labels
        self.device = device
        # 🎯 训练时应用频谱增强
        self.augmentor = CombinedAugmentation(
            sample_rate=Config.SAMPLE_RATE, 
            device=device,
            deterministic=False,  # 训练时用随机增强
            audio_augment=False,
            spec_augment=True     # 明确启用频谱增强
        )
        
        print(f"📊 加载预计算训练特征: {self.features.shape}")

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        x = torch.tensor(self.features[idx], dtype=torch.float32)
        
        # 确保输入是3D: [1, n_mels, time]
        x = self._ensure_3d(x)
        
        # 🎯 应用频谱增强（在GPU上）
        x = self.augmentor(x)
        
        y = int(self.labels[idx])
        return x, y
    
    def _ensure_3d(self, x):
        """确保输入为3D张量"""
        if x.ndim == 4:
            if x.shape[0] == 1 and x.shape[1] == 1:
                x = x.squeeze(0)
            else:
                x = x[0]
        if x.ndim == 2:
            x = x.unsqueeze(0)
        return x

class PrecomputedEvalDataset(Dataset):
    """预计算评估数据集（无增强）"""
    
    def __init__(self, features, labels, device):  # 🎯 修复：直接传入数组
        self.features = features
        self.labels = labels
        self.device = device
        print(f"📊 加载预计算评估特征: {self.features.shape}")

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        x = torch.tensor(self.features[idx], dtype=torch.float32)
        x = self._ensure_3d(x)
        y = int(self.labels[idx])
        return x, y
    
    def _ensure_3d(self, x):
        """确保输入为3D张量"""
        if x.ndim == 4:
            if x.shape[0] == 1 and x.shape[1] == 1:
                x = x.squeeze(0)
            else:
                x = x[0]
        if x.ndim == 2:
            x = x.unsqueeze(0)
        return x

# ==================== 数据加载器 ====================
class DataLoaderFactory:
    """数据加载器工厂"""
    
    @staticmethod
    def create_loader(dataset, batch_size, shuffle=True):
        return DataLoader(
            dataset, 
            batch_size=batch_size, 
            shuffle=shuffle, 
            drop_last=True, 
            num_workers=0,  # 单进程
            pin_memory=True  # 加速GPU数据传输
        )

# ==================== 训练模块 ====================
class ModelTrainer:
    """模型训练器"""
    
    def __init__(self, model, train_loader, val_loader, test_loader):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.device = Config.DEVICE
        
    def train_single_epoch(self, criterion, optimizer, scaler, scheduler, epoch_desc):
        """训练单个epoch"""

        # PyTorch的切换到训练的指令
        self.model.train()
        total_loss, correct, total = 0, 0, 0
        
        batch_pbar = tqdm(self.train_loader, desc=epoch_desc, leave=False)

        for batch_idx, (x, y) in enumerate(batch_pbar):
            x = x.to(self.device, non_blocking=True)
            y = y.to(self.device, non_blocking=True)
            
            optimizer.zero_grad()
            with torch.amp.autocast(device_type='cuda', dtype=torch.float16):
                pred = self.model(x)
                loss = criterion(pred, y)
            
            # mixed precision grad management
            scaler.scale(loss).backward()   # grad squeeze -> transfer(auto finish in GPU)
            scaler.step(optimizer)          # unsqueeze
            scaler.update()                 # update
            
            # 学习率调度
            if scheduler is not None:
                current_lr = scheduler.get_last_lr()[0] 
                scheduler.step()
            
            # 统计信息
            total_loss += loss.item()   # loss_value = loss.item()  total_loss += loss_value
            _, predicted = pred.max(dim=1)
            batch_correct = predicted.eq(y).sum().item()
            correct += batch_correct
            total += y.size(dim=0)
            
            # 更新进度条
            batch_acc = 100 * batch_correct / y.size(0)
            current_lr = current_lr if scheduler else optimizer.param_groups[0]['lr']
            # instant print
            batch_pbar.set_postfix({
                'Loss': f'{loss.item():.4f}',
                'BatchAcc': f'{batch_acc:.1f}%',
                'LR': f'{current_lr:.2e}'
            })
        
        return total_loss / len(self.train_loader), 100 * correct / total
    
    def validate(self, criterion):
        """验证模型"""
        self.model.eval()
        total_loss, correct, total = 0, 0, 0
        
        with torch.no_grad():
            for x, y in self.val_loader:
                x = x.to(self.device, non_blocking=True)
                y = y.to(self.device, non_blocking=True)
                
                with torch.amp.autocast(device_type='cuda', dtype=torch.float16):
                    pred = self.model(x)
                    loss = criterion(pred, y)
                
                total_loss += loss.item()
                _, predicted = pred.max(dim=1)
                correct += predicted.eq(y).sum().item()
                total += y.size(dim=0)
        
        return total_loss / len(self.val_loader), 100 * correct / total
    
    def test(self, criterion):
        """测试模型"""
        self.model.eval()
        total_loss, correct, total = 0, 0, 0
        
        with torch.no_grad():
            for x, y in self.test_loader:
                x = x.to(self.device, non_blocking=True)
                y = y.to(self.device, non_blocking=True)
                
                with torch.amp.autocast(device_type='cuda', dtype=torch.float16):
                    pred = self.model(x)
                    loss = criterion(pred, y)
                
                total_loss += loss.item()
                _, predicted = pred.max(1)
                correct += predicted.eq(y).sum().item()
                total += y.size(0)
        
        return total_loss / len(self.test_loader), 100 * correct / total

    def train(self, epochs=120, patience=10):
        """完整训练流程"""
        # 损失函数计算-基于成熟的交叉熵理论
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.AdamW(
                                    self.model.parameters(), 
                                    lr=Config.LEARNING_RATE, 
                                    weight_decay=Config.WEIGHT_DECAY
                                    )
        scaler = torch.amp.GradScaler()
        
        # 学习率调度器
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=Config.LEARNING_RATE * Config.MAX_LR_FACTOR,
            epochs=epochs,
            steps_per_epoch=len(self.train_loader),
            pct_start=Config.WARMUP_RATIO,
            div_factor=Config.INIT_DIV_FACTOR,
            final_div_factor=Config.FINAL_DIV_FACTOR
        )

        best_acc, epochs_no_improve = 0, 0
        
        print(f"🚀 开始训练 | Epochs: {epochs} | 学习率: {Config.LEARNING_RATE}")
        print(f"🔥 使用OneCycleLR: warmup={Config.WARMUP_RATIO*100}%")
        print(f"🎯 增强策略: 预计算音频增强特征 + 训练时实时频谱增强")

        epoch_pbar = tqdm(range(epochs), desc="训练进度", leave=True)
        
        for epoch in epoch_pbar:
            epoch_desc = f"Epoch {epoch+1:02d}/{epochs}"
            
            # 训练
            train_loss, train_acc = self.train_single_epoch(
                criterion, optimizer, scaler, scheduler, epoch_desc
            )
            
            # 验证和测试
            val_loss, val_acc = self.validate(criterion)
            test_loss, test_acc = self.test(criterion)
            
            # 更新进度条
            current_lr = scheduler.get_last_lr()[0]
            epoch_pbar.set_postfix({
                'Train': f'{train_acc:.1f}%',
                'Val': f'{val_acc:.1f}%', 
                'Test': f'{test_acc:.1f}%',
                'LR': f'{current_lr:.2e}'
            })
            
            # 打印详细结果
            print(f"\nEpoch {epoch+1:02d}: Train {train_acc:.2f}% | Val {val_acc:.2f}% | Test {test_acc:.2f}% | LR {current_lr:.2e} | LOSS {train_loss:.4f}")
            
            # 保存最佳模型
            if val_acc > best_acc:
                best_acc = val_acc
                epochs_no_improve = 0
                torch.save(self.model.state_dict(), Config.BEST_MODEL_PATH)
                print(f"💾 保存最佳模型 | Val Acc: {best_acc:.2f}%")
            else:
                epochs_no_improve += 1
                if epochs_no_improve >= patience:
                    print(f"🛑 早停触发 | 最佳验证准确率: {best_acc:.2f}%")
                    break
                
        return Config.BEST_MODEL_PATH

# ==================== 主程序 ====================
def main():
    # 初始化配置
    print("🎯 初始化配置...")
    label_mapping = {tag: idx for idx, tag in enumerate(Config.TOP10_TAGS)}
    num_classes = len(label_mapping)
    print(f"🔢 标签映射: {label_mapping}")
    
    # 清理并创建预计算目录
    if os.path.exists(Config.PRECOMPUTE_DIR):
        shutil.rmtree(Config.PRECOMPUTE_DIR)
        print("🗑️ 已删除旧的预计算文件")
    os.makedirs(Config.PRECOMPUTE_DIR, exist_ok=True)
    print("📁 创建新的预计算目录")
    
    # 🎯 创建完整数据集（音频增强）
    print("🔧 创建完整数据集...")
    full_dataset = EmotionSoundDataset(
        annotations_file=Config.ANNOTATIONS_FILE,
        audio_dir=Config.AUDIO_DIR,
        target_sample_rate=Config.SAMPLE_RATE,
        num_samples=Config.NUM_SAMPLES,
        device='cpu',
        use_log_mel=True,
        top_n_classes=10,
        augment=True,                            # ✅ 启用增强总开关
        audio_augment=True,                      # ✅ 启用音频增强
        spec_augment=False,
        deterministic_augment=True,
        augment_seed=42
    )
    
    # 预计算完整数据集
    full_features_path, full_labels_path = FeaturePrecomputer.precompute_features(
        full_dataset, 
        Config.PRECOMPUTE_DIR, 
        "full"
    )

    # 🎯 加载预计算数据
    full_features = np.load(full_features_path)
    full_labels = np.load(full_labels_path)
    print(f"📊 总样本数: {len(full_labels)}")

    # 分层抽样划分
    print("📊 进行分层抽样划分...")
    
    # 第一次划分：训练集 vs (验证+测试集)
    X_train, X_temp, y_train, y_temp = train_test_split(
        full_features, 
        full_labels, 
        test_size=0.3, 
        stratify=full_labels,
        random_state=42
    )
    
    # 第二次划分：验证集 vs 测试集
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, 
        y_temp,
        test_size=0.5,
        stratify=y_temp,
        random_state=42
    )
    
    print(f"✅ 分层抽样完成: 训练集 {len(y_train)} | 验证集 {len(y_val)} | 测试集 {len(y_test)}")
    
    # 🎯 创建最终数据集 - 直接传入划分后的数组
    print("🔧 创建最终数据集...")
    train_dataset = PrecomputedTrainDataset(X_train, y_train, Config.DEVICE)
    val_dataset = PrecomputedEvalDataset(X_val, y_val, Config.DEVICE)
    test_dataset = PrecomputedEvalDataset(X_test, y_test, Config.DEVICE)

    print(f"🎯 最终数据集: 训练集 {len(train_dataset)} | 验证集 {len(val_dataset)} | 测试集 {len(test_dataset)}")

    # 🎯 数据验证
    print("=== 数据划分验证 ===")
    print(f"训练集样本数: {len(train_dataset)}")
    print(f"验证集样本数: {len(val_dataset)}") 
    print(f"测试集样本数: {len(test_dataset)}")
    print(f"训练集标签分布: {np.bincount(train_dataset.labels)}")
    print(f"验证集标签分布: {np.bincount(val_dataset.labels)}")
    print(f"测试集标签分布: {np.bincount(test_dataset.labels)}")

    # 创建数据加载器
    train_loader = DataLoaderFactory.create_loader(train_dataset, Config.BATCH_SIZE, shuffle=True)
    val_loader = DataLoaderFactory.create_loader(val_dataset, Config.BATCH_SIZE, shuffle=False)
    test_loader = DataLoaderFactory.create_loader(test_dataset, Config.BATCH_SIZE, shuffle=False)
    
    # 测试数据形状
    for x, y in train_loader:
        print(f"🔍 训练集Batch形状: 输入 {x.shape} | 标签 {y.shape}")
        break
    
    # 初始化模型
    model = CNNNetwork10(num_classes=num_classes).to(Config.DEVICE)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"🤖 使用改进版模型 | 参数量: {total_params} ({total_params/1e6:.2f}M)")
    print(f"🎯 优化方案: 预计算音频增强特征 + 训练时实时频谱增强")
    
    # 开始训练
    trainer = ModelTrainer(model, train_loader, val_loader, test_loader)
    best_model_path = trainer.train(Config.EPOCHS, Config.PATIENCE)
    
    # 最终测试
    model.load_state_dict(torch.load(best_model_path))
    final_test_loss, final_test_acc = trainer.test(nn.CrossEntropyLoss())
    print(f"🎉 最终测试准确率: {final_test_acc:.2f}%")
    print(f"💾 最佳模型已保存: {best_model_path}")

if __name__ == "__main__":
    main()