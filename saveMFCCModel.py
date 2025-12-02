import os
import numpy as np
import librosa
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.model_selection import cross_val_score
from joblib import dump, load

def extract_mfcc(file_path, max_pad_len=860, duration=10):
    """
    Extract MFCC features from the first `duration` seconds of an audio file.
    """
    try:
        audio, sample_rate = librosa.load(file_path, sr=22050, mono=True, duration=duration)
        mfccs = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=13)

        pad_width = max_pad_len - mfccs.shape[1]
        if pad_width > 0:
            mfccs = np.pad(mfccs, pad_width=((0, 0), (0, pad_width)), mode='constant')
        else:
            mfccs = mfccs[:, :max_pad_len]

        return mfccs.flatten()

    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return None

def load_dataset(data_path):
    """
    Load dataset and extract MFCC features.
    """
    features, labels = [], []
    for label, class_name in enumerate(['crackling', 'non_crackling']):
        class_path = os.path.join(data_path, class_name)
        for filename in os.listdir(class_path):
            file_path = os.path.join(class_path, filename)
            mfcc_features = extract_mfcc(file_path)
            if mfcc_features is not None:
                features.append(mfcc_features)
                labels.append(label)

    return np.array(features), np.array(labels)

def train_and_save_model(train_path, test_path, model_path, scaler_path, pca_path):
    """
    Train SVM classifier, evaluate performance using cross-validation, and save the model, scaler, and PCA.
    """
    # Load training and testing datasets separately
    X_train, y_train = load_dataset(train_path)
    X_test, y_test = load_dataset(test_path)

    # Check if datasets are empty
    if X_train.size == 0 or X_test.size == 0:
        raise ValueError("Training or testing dataset is empty. Ensure valid audio files are present.")

    # Standardize features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Apply PCA for dimensionality reduction
    pca = PCA(n_components=50)  # Adjust as needed
    X_train_pca = pca.fit_transform(X_train_scaled)
    X_test_pca = pca.transform(X_test_scaled)

    # Train SVM classifier
    svm_classifier = SVC(kernel='rbf', probability=True)
    svm_classifier.fit(X_train_pca, y_train)

    # Cross-validation on training data
    scores = cross_val_score(svm_classifier, X_train_pca, y_train, cv=5)
    print(f"Cross-validation Accuracy: {scores.mean():.4f} ± {scores.std():.4f}")

    # Evaluate on training data
    y_train_pred = svm_classifier.predict(X_train_pca)
    print("Training Performance:")
    print(f"Accuracy: {accuracy_score(y_train, y_train_pred):.4f}")
    print("Classification Report:\n", classification_report(y_train, y_train_pred, target_names=['Non-Crackling', 'Crackling']))
    print("Confusion Matrix:\n", confusion_matrix(y_train, y_train_pred))

    # Evaluate on testing data
    y_test_pred = svm_classifier.predict(X_test_pca)
    print("\nTesting Performance:")
    print(f"Accuracy: {accuracy_score(y_test, y_test_pred):.4f}")
    print("Classification Report:\n", classification_report(y_test, y_test_pred, target_names=['Non-Crackling', 'Crackling']))
    print("Confusion Matrix:\n", confusion_matrix(y_test, y_test_pred))

    # Save the model, scaler, and PCA
    dump(svm_classifier, model_path, compress=0)
    dump(scaler, scaler_path, compress=0)
    dump(pca, pca_path, compress=0)
    print(f"\nModel saved to {model_path}")
    print(f"Scaler saved to {scaler_path}")
    print(f"PCA saved to {pca_path}")

# Train and save
if __name__ == "__main__":
    train_path = '/Users/manoj/Documents/PYTHON/CODE_FY/data/wav/train'
    test_path = '/Users/manoj/Documents/PYTHON/CODE_FY/data/wav/test'
    model_path = 'svm_model.pkl'
    scaler_path = 'scaler.pkl'
    pca_path = 'pca.pkl'
    train_and_save_model(train_path, test_path, model_path, scaler_path, pca_path)
