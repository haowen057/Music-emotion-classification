import os
import librosa
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, classification_report
from tqdm import tqdm
from scipy.stats import skew, kurtosis
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

FEATURE_FILE = "features.npz"

def extract_features(file_path):
    try:
        y, sr = librosa.load(file_path, sr=None)
    except Exception as e:
        print(f"Fail: {file_path} | {e}")
        return None

    if len(y) < sr * 10:
        y = np.pad(y, (0, sr*10 - len(y)))
    else:
        y = y[:sr*10]

    feats = []

    zcr = librosa.feature.zero_crossing_rate(y)
    feats.extend([np.mean(zcr), np.std(zcr)])

    rms = librosa.feature.rms(y=y)
    feats.extend([np.mean(rms), np.std(rms)])

    sc = librosa.feature.spectral_centroid(y=y, sr=sr)
    feats.extend([np.mean(sc), np.std(sc)])

    sbw = librosa.feature.spectral_bandwidth(y=y, sr=sr)
    feats.extend([np.mean(sbw), np.std(sbw)])

    contrast = librosa.feature.spectral_contrast(y=y, sr=sr)
    feats.extend([np.mean(contrast), np.std(contrast)])

    flatness = librosa.feature.spectral_flatness(y=y)
    feats.extend([np.mean(flatness), np.std(flatness)])

    rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
    feats.extend([np.mean(rolloff), np.std(rolloff)])

    feats.extend([skew(y), kurtosis(y)])

    # --- MFCC mean + delta mean ---
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    delta_mfcc = librosa.feature.delta(mfcc)
    feats.extend(np.mean(mfcc, axis=1))
    feats.extend(np.mean(delta_mfcc, axis=1))

    arr = np.array(feats, dtype=np.float32)
    arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
    return arr

def load_dataset(root_dir):
    X, y = [], []
    for label_name in sorted(os.listdir(root_dir)):
        label_dir = os.path.join(root_dir, label_name)
        if not os.path.isdir(label_dir):
            continue
        for file_name in tqdm(os.listdir(label_dir), desc=f"Processing {label_name}"):
            if not file_name.endswith(".wav"):
                continue
            file_path = os.path.join(label_dir, file_name)
            feats = extract_features(file_path)
            if feats is None:
                continue
            X.append(feats)
            y.append(label_name)
    return np.array(X), np.array(y)

def plot_confusion_matrix(predictions, targets, class_mapping):
    """
    Plot confusion matrix (English version)
    """
    cm = confusion_matrix(targets, predictions)

    plt.figure(figsize=(12, 10))
    sns.heatmap(
        cm, annot=True, fmt='d', cmap='Blues',
        xticklabels=class_mapping,
        yticklabels=class_mapping
    )
    plt.title('Confusion Matrix - Music Mood Classification', fontsize=16, fontweight='bold')
    plt.xlabel('Predicted Class', fontsize=13)
    plt.ylabel('True Class', fontsize=13)
    plt.xticks(rotation=45, ha='right', fontsize=11)
    plt.yticks(rotation=0, fontsize=11)
    plt.tight_layout()
    plt.show()



if __name__ == "__main__":
    dataset_dir = "mood"

    if os.path.exists(FEATURE_FILE):
        data = np.load(FEATURE_FILE)
        X = data['X']
        y = data['y']
    else:
        X, y = load_dataset(dataset_dir)
        np.savez(FEATURE_FILE, X=X, y=y)
        print(f"Saved {FEATURE_FILE}")

    encoder = LabelEncoder()
    y_encoded = encoder.fit_transform(y)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    print("NaN check:", np.isnan(X_scaled).sum())

    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y_encoded, test_size=0.1, random_state=42, stratify=y_encoded
    )


    model = MLPClassifier(
        hidden_layer_sizes=(512, 256, 128),
        activation='relu',
        solver='adam',
        batch_size=64,
        learning_rate_init=1e-4,
        max_iter=500,
        early_stopping=False,
        random_state=42,
        verbose=True
    )

    print("Training MLP...")
    model.fit(X_train, y_train)

    test_pred = model.predict(X_test)
    test_acc = accuracy_score(y_test, test_pred)
    print(f"Accuracy: {test_acc:.4f}")

    print("\nreport:")
    print(classification_report(y_test, test_pred, target_names=encoder.classes_, digits=4))

    plot_confusion_matrix(
        predictions=test_pred,
        targets=y_test,
        class_mapping=encoder.classes_
    )
