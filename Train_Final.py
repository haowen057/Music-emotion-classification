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


class Config:
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    
    BATCH_SIZE = 64
    EPOCHS = 120
    LEARNING_RATE = 1e-3
    WEIGHT_DECAY = 1e-4
    PATIENCE = 10
    
    SAMPLE_RATE = 22050
    NUM_SAMPLES = SAMPLE_RATE * 10
    
    WARMUP_RATIO = 0.1
    MAX_LR_FACTOR = 1.5
    INIT_DIV_FACTOR = 25
    FINAL_DIV_FACTOR = 1e4
    
    BEST_MODEL_PATH = "best_emotion_classifier_amp_enhanced_11.pth"
    ANNOTATIONS_FILE = r"C:\Users\14217\Desktop\DT2470\Predictions with sound classifier\metadata_balanced_10s_multi.tsv"
    AUDIO_DIR = r"C:\Users\14217\Desktop\DT2470\Predictions with sound classifier"
    PRECOMPUTE_DIR = os.path.join(AUDIO_DIR, "precomputed")
    
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

class FeaturePrecomputer:
    @staticmethod
    def precompute_features(dataset, save_dir, suffix=""):
        features = []
        labels = []
        
        augment_info = dataset.get_dataset_info()
        print(f"Precomputing features | Audio augmentation: {augment_info['audio_augment']} | Spectrogram augmentation: {augment_info['spec_augment']}")
        
        for i in tqdm(range(len(dataset))):
            try:
                mel, label = dataset[i]
                
                if mel.ndim == 2:
                    mel = mel.unsqueeze(0)
                elif mel.ndim == 3 and mel.shape[0] != 1:
                    mel = mel.mean(0, keepdim=True)
                
                features.append(mel.cpu().detach().numpy())
                labels.append(label)
                
            except Exception as e:
                print(f"Skipping sample {i}: {e}")
                continue
            
        features_array = np.stack(features).astype(np.float32)
        labels_array = np.array(labels)
    
        features_path = os.path.join(save_dir, f"features_{suffix}.npy")
        labels_path = os.path.join(save_dir, f"labels_{suffix}.npy")
        
        np.save(features_path, features_array)
        np.save(labels_path, labels_array)
        
        print(f"Precomputation completed: {len(features_array)} samples")
        return features_path, labels_path

class PrecomputedTrainDataset(Dataset):
    def __init__(self, features, labels, device):
        self.features = features
        self.labels = labels
        self.device = device
        self.augmentor = CombinedAugmentation(
            sample_rate=Config.SAMPLE_RATE, 
            device=device,
            deterministic=False,
            audio_augment=False,
            spec_augment=True
        )
        
        print(f"Loading precomputed training features: {self.features.shape}")

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        x = torch.tensor(self.features[idx], dtype=torch.float32)
        
        x = self._ensure_3d(x)
        
        x = self.augmentor(x)
        
        y = int(self.labels[idx])
        return x, y
    
    def _ensure_3d(self, x):
        if x.ndim == 4:
            if x.shape[0] == 1 and x.shape[1] == 1:
                x = x.squeeze(0)
            else:
                x = x[0]
        if x.ndim == 2:
            x = x.unsqueeze(0)
        return x

class PrecomputedEvalDataset(Dataset):
    def __init__(self, features, labels, device):
        self.features = features
        self.labels = labels
        self.device = device
        print(f"Loading precomputed evaluation features: {self.features.shape}")

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        x = torch.tensor(self.features[idx], dtype=torch.float32)
        x = self._ensure_3d(x)
        y = int(self.labels[idx])
        return x, y
    
    def _ensure_3d(self, x):
        if x.ndim == 4:
            if x.shape[0] == 1 and x.shape[1] == 1:
                x = x.squeeze(0)
            else:
                x = x[0]
        if x.ndim == 2:
            x = x.unsqueeze(0)
        return x

class DataLoaderFactory:
    @staticmethod
    def create_loader(dataset, batch_size, shuffle=True):
        return DataLoader(
            dataset, 
            batch_size=batch_size, 
            shuffle=shuffle, 
            drop_last=True, 
            num_workers=0,
            pin_memory=True
        )

class ModelTrainer:
    def __init__(self, model, train_loader, val_loader, test_loader):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.device = Config.DEVICE
        
    def train_single_epoch(self, criterion, optimizer, scaler, scheduler, epoch_desc):
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
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            if scheduler is not None:
                current_lr = scheduler.get_last_lr()[0] 
                scheduler.step()
            
            total_loss += loss.item()
            _, predicted = pred.max(dim=1)
            batch_correct = predicted.eq(y).sum().item()
            correct += batch_correct
            total += y.size(dim=0)
            
            batch_acc = 100 * batch_correct / y.size(0)
            current_lr = current_lr if scheduler else optimizer.param_groups[0]['lr']
            batch_pbar.set_postfix({
                'Loss': f'{loss.item():.4f}',
                'BatchAcc': f'{batch_acc:.1f}%',
                'LR': f'{current_lr:.2e}'
            })
        
        return total_loss / len(self.train_loader), 100 * correct / total
    
    def validate(self, criterion):
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
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.AdamW(
            self.model.parameters(), 
            lr=Config.LEARNING_RATE, 
            weight_decay=Config.WEIGHT_DECAY
        )
        scaler = torch.amp.GradScaler()
        
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
        
        print(f"Starting training | Epochs: {epochs} | Learning rate: {Config.LEARNING_RATE}")
        print(f"Using OneCycleLR: warmup={Config.WARMUP_RATIO*100}%")
        print(f"Augmentation strategy: Precomputed audio augmented features + Real-time spectrogram augmentation during training")

        epoch_pbar = tqdm(range(epochs), desc="Training progress", leave=True)
        
        for epoch in epoch_pbar:
            epoch_desc = f"Epoch {epoch+1:02d}/{epochs}"
            
            train_loss, train_acc = self.train_single_epoch(
                criterion, optimizer, scaler, scheduler, epoch_desc
            )
            
            val_loss, val_acc = self.validate(criterion)
            test_loss, test_acc = self.test(criterion)
            
            current_lr = scheduler.get_last_lr()[0]
            epoch_pbar.set_postfix({
                'Train': f'{train_acc:.1f}%',
                'Val': f'{val_acc:.1f}%', 
                'Test': f'{test_acc:.1f}%',
                'LR': f'{current_lr:.2e}'
            })
            
            print(f"\nEpoch {epoch+1:02d}: Train {train_acc:.2f}% | Val {val_acc:.2f}% | Test {test_acc:.2f}% | LR {current_lr:.2e} | LOSS {train_loss:.4f}")
            
            if val_acc > best_acc:
                best_acc = val_acc
                epochs_no_improve = 0
                torch.save(self.model.state_dict(), Config.BEST_MODEL_PATH)
                print(f"Saved best model | Val Acc: {best_acc:.2f}%")
            else:
                epochs_no_improve += 1
                if epochs_no_improve >= patience:
                    print(f"Early stopping triggered | Best validation accuracy: {best_acc:.2f}%")
                    break
                
        return Config.BEST_MODEL_PATH

def main():
    print("Initializing configuration...")
    label_mapping = {tag: idx for idx, tag in enumerate(Config.TOP10_TAGS)}
    num_classes = len(label_mapping)
    print(f"Label mapping: {label_mapping}")
    
    if os.path.exists(Config.PRECOMPUTE_DIR):
        shutil.rmtree(Config.PRECOMPUTE_DIR)
        print("Deleted old precomputed files")
    os.makedirs(Config.PRECOMPUTE_DIR, exist_ok=True)
    print("Created new precomputation directory")
    
    print("Creating full dataset...")
    full_dataset = EmotionSoundDataset(
        annotations_file=Config.ANNOTATIONS_FILE,
        audio_dir=Config.AUDIO_DIR,
        target_sample_rate=Config.SAMPLE_RATE,
        num_samples=Config.NUM_SAMPLES,
        device='cpu',
        use_log_mel=True,
        top_n_classes=10,
        augment=True,
        audio_augment=True,
        spec_augment=False,
        deterministic_augment=True,
        augment_seed=42
    )
    
    full_features_path, full_labels_path = FeaturePrecomputer.precompute_features(
        full_dataset, 
        Config.PRECOMPUTE_DIR, 
        "full"
    )

    full_features = np.load(full_features_path)
    full_labels = np.load(full_labels_path)
    print(f"Total samples: {len(full_labels)}")

    print("Performing stratified sampling...")
    
    X_train, X_temp, y_train, y_temp = train_test_split(
        full_features, 
        full_labels, 
        test_size=0.3, 
        stratify=full_labels,
        random_state=42
    )
    
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, 
        y_temp,
        test_size=0.5,
        stratify=y_temp,
        random_state=42
    )
    
    print(f"Stratified sampling completed: Training set {len(y_train)} | Validation set {len(y_val)} | Test set {len(y_test)}")
    
    print("Creating final datasets...")
    train_dataset = PrecomputedTrainDataset(X_train, y_train, Config.DEVICE)
    val_dataset = PrecomputedEvalDataset(X_val, y_val, Config.DEVICE)
    test_dataset = PrecomputedEvalDataset(X_test, y_test, Config.DEVICE)

    print(f"Final datasets: Training set {len(train_dataset)} | Validation set {len(val_dataset)} | Test set {len(test_dataset)}")

    print("=== Data Split Validation ===")
    print(f"Training set samples: {len(train_dataset)}")
    print(f"Validation set samples: {len(val_dataset)}") 
    print(f"Test set samples: {len(test_dataset)}")
    print(f"Training set label distribution: {np.bincount(train_dataset.labels)}")
    print(f"Validation set label distribution: {np.bincount(val_dataset.labels)}")
    print(f"Test set label distribution: {np.bincount(test_dataset.labels)}")

    train_loader = DataLoaderFactory.create_loader(train_dataset, Config.BATCH_SIZE, shuffle=True)
    val_loader = DataLoaderFactory.create_loader(val_dataset, Config.BATCH_SIZE, shuffle=False)
    test_loader = DataLoaderFactory.create_loader(test_dataset, Config.BATCH_SIZE, shuffle=False)
    
    for x, y in train_loader:
        print(f"Training batch shape: Input {x.shape} | Labels {y.shape}")
        break
    
    model = CNNNetwork10(num_classes=num_classes).to(Config.DEVICE)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Using improved model | Parameters: {total_params} ({total_params/1e6:.2f}M)")
    print(f"Optimization strategy: Precomputed audio augmented features + Real-time spectrogram augmentation during training")
    
    trainer = ModelTrainer(model, train_loader, val_loader, test_loader)
    best_model_path = trainer.train(Config.EPOCHS, Config.PATIENCE)
    
    model.load_state_dict(torch.load(best_model_path))
    final_test_loss, final_test_acc = trainer.test(nn.CrossEntropyLoss())
    print(f"Final test accuracy: {final_test_acc:.2f}%")
    print(f"Best model saved: {best_model_path}")

if __name__ == "__main__":
    main()

