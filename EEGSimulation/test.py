import os
import numpy as np
import mne
from tensorflow.keras.models import load_model
from sklearn.preprocessing import StandardScaler
from scipy.stats import entropy, mode
import scipy.signal as signal
import scipy.integrate as integrate

# Paramètres de segmentation et du modèle
segment_length_sec = 5  # Durée du segment en secondes
sampling_rate = 500  # Fréquence d'échantillonnage
segment_length = segment_length_sec * sampling_rate
overlap_ratio = 0.5  # Chevauchement de 50%
overlap_step = int(segment_length * (1 - overlap_ratio))

# Bande de fréquence pour extraire les caractéristiques
frequency_bands = {
    'delta': (0.5, 4),
    'theta': (4, 8),
    'alpha': (8, 13),
    'beta': (13, 25),
    'gamma': (25, 45)
}
sampling_rate_features = 128  # Fréquence d'échantillonnage pour les bandes de fréquence

# Charger les modèles
model_path = "/Users/pro/Desktop/eeg_model_for5s.h5"
autoencoder_path = "/Users/pro/Desktop/eeg_autoencoder.h5"

model = load_model(model_path)
encoder = load_model(autoencoder_path)

# Fonction pour extraire les bandes de fréquence
def extract_band_power(segments, frequency_bands, sampling_rate):
    band_powers = []
    for segment in segments:
        freqs, psd = signal.welch(segment, fs=sampling_rate, nperseg=len(segment))
        band_power = []
        for band, (low_freq, high_freq) in frequency_bands.items():
            idx_band = np.logical_and(freqs >= low_freq, freqs <= high_freq)
            power = integrate.simps(psd[idx_band], freqs[idx_band]) if np.any(idx_band) else 0
            band_power.append(power)
        band_powers.append(band_power)
    return np.array(band_powers)

# Fonction pour calculer l'entropie spectrale
def calculate_spectral_entropy(band_powers):
    normalized_powers = band_powers / np.sum(band_powers, axis=1, keepdims=True)
    spectral_entropy = entropy(normalized_powers, axis=1)
    return spectral_entropy

# Fonction principale pour traiter un fichier EEG
from scipy.stats import mode

from scipy.stats import mode

def process_test_file(eeg_file_path):
    print(f"Processing test file: {eeg_file_path}")

    # Charger les données EEG
    raw = mne.io.read_raw_eeglab(eeg_file_path, preload=True)
    data = raw.get_data()  # (n_channels, n_samples)
    n_channels, n_samples = data.shape

    # Segmenter avec chevauchement
    data_segments = []
    start = 0
    while start + segment_length <= n_samples:
        end = start + segment_length
        segment = data[:, start:end]
        data_segments.append(segment)
        start += overlap_step

    data_segments = np.array(data_segments)
    print(f"Segments created: {data_segments.shape}")

    # Standardiser les segments
    scaler = StandardScaler()
    standardized_segments = np.array([scaler.fit_transform(segment) for segment in data_segments])

    # Encoder les segments avec l'autoencodeur
    encoded_segments = encoder.predict(standardized_segments)
    print(f"Encoded segments: {encoded_segments.shape}")

    # Extraire les bandes de fréquence
    band_powers = extract_band_power(encoded_segments, frequency_bands, sampling_rate_features)
    spectral_entropy = calculate_spectral_entropy(band_powers)

    # Combiner les caractéristiques
    combined_features = np.hstack((band_powers, spectral_entropy.reshape(-1, 1)))
    print(f"Combined features shape: {combined_features.shape}")

    # Créer des séquences pour le modèle LSTM
    sequence_length = 10
    sequences = []
    for i in range(len(combined_features) - sequence_length):
        sequence = combined_features[i:i + sequence_length]
        sequences.append(sequence)

    sequences = np.array(sequences)
    print(f"Sequences shape: {sequences.shape}")

    # Faire des prédictions avec le modèle
    predictions = model.predict(sequences)
    predicted_labels = np.argmax(predictions, axis=1)

    if len(predicted_labels) == 0:
        print("No predictions were made.")
        return

    # Approche 1 : Classe majoritaire
    mode_result = mode(predicted_labels, keepdims=True)  # Assurez-vous que keepdims=True
    if mode_result.mode.size > 0:
        final_prediction_majority = mode_result.mode[0]
    else:
        final_prediction_majority = None

    # Approche 2 : Moyenne des probabilités
    if predictions.size > 0:
        average_probabilities = np.mean(predictions, axis=0)
        final_prediction_avg = np.argmax(average_probabilities)
    else:
        final_prediction_avg = None

    print(f"Predicted class for each segment: {predicted_labels}")
    print(f"Final predicted class (majority): {final_prediction_majority}")
    print(f"Final predicted class (average probabilities): {final_prediction_avg}")

if __name__ == "__main__":

    # Tester le deuxième fichier EEG
    test_file_2 = "/Users/pro/Desktop/BigData/ProjetEEG/ds004504-1.0.7/derivatives/sub-082/eeg/sub-082_task-eyesclosed_eeg.set"
    process_test_file(test_file_2)
