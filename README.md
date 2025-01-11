# 🧠 EEG Signal Classification: Alzheimer's Disease, Frontotemporal Dementia, and Cognitively Normal  

## 📜 Overview  
This project focuses on classifying resting-state EEG signals into three categories:  
- **Alzheimer's Disease (AD)**  
- **Frontotemporal Dementia (FTD)**  
- **Cognitively Normal (CN)**  

The project employs advanced preprocessing techniques and a deep learning model architecture combining an **Autoencoder** and a **Bidirectional Long Short-Term Memory (LSTM)** network to achieve high accuracy in classification.  

---

## 🌟 Key Features  

### 📊 Participants  
- **AD Group**: 36 subjects  
- **FTD Group**: 23 subjects  
- **CN Group**: 29 subjects  

### ⚙️ Preprocessing  
- **Band-pass filter**: 0.5–45 Hz  
- **Artifact removal**:  
  - Independent Component Analysis (ICA)  
  - Artifact Subspace Reconstruction (ASR)  
- **Eye and jaw artifacts rejection**  

### 🧪 Data  
- **EEG recordings**: 19 channels  
- **Sampling rate**: 500 Hz  
- **Recording durations**:  
  - AD: ~485 minutes  
  - FTD: ~276 minutes  
  - CN: ~402 minutes  

---

## 📚 Dataset Overview  

### 📜 Dataset Description  
This project utilizes an **open dataset from OpenNeuro** containing resting-state EEG recordings from 88 participants classified into three groups:  
1. **Alzheimer's Disease (AD)**  
2. **Frontotemporal Dementia (FTD)**  
3. **Cognitively Normal (CN)**  

### 👥 Participants Information  
- **Distribution of participants**:  
  - AD: 36 subjects  
  - FTD: 23 subjects  
  - CN: 29 subjects  
- **Average MMSE scores by group** (Mini-Mental State Examination):  
  - AD: ~19.4  
  - FTD: ~23.7  
  - CN: ~29.5  
- **Average age by group**:  
  - AD: ~72 years  
  - FTD: ~66 years  
  - CN: ~60 years  

### ⚙️ Recording Details  
- **EEG Device**: Nihon Kohden EEG 2100  
- **Number of scalp electrodes**: 19  
- **Sampling rate**: 500 Hz  
- **Average recording durations**:  
  - AD: ~13.5 minutes  
  - FTD: ~12 minutes  
  - CN: ~13.8 minutes  

### 🔧 Preprocessing Steps  
1. **Band-pass filtering**: 0.5–45 Hz.  
2. **Artifact removal techniques**:  
   - Artifact Subspace Reconstruction (ASR).  
   - Independent Component Analysis (ICA).  
3. **Re-referencing**: EEG signals were re-referenced to the average of electrodes A1 and A2.  

---

## 🏗️ Model Architecture and Training  

### 🌟 Feature Extraction with Autoencoder  
1. **Data Segmentation**:  
   - EEG signals were segmented into overlapping epochs of **5 seconds** with a **50% overlap**.  
2. **Autoencoder**:  
   - **Encoder**: Dense layers with ReLU activations to extract compressed representations.  
   - **Decoder**: Reconstructs the original signal from the compressed representation.  

The autoencoder provides a robust feature representation of the EEG data, enabling efficient and accurate classification.  

---

### 🔄 Sequence Learning with Bidirectional LSTM  
1. **Sequence Creation**:  
   - Autoencoder-extracted features combined with **spectral entropy** values are aggregated into feature vectors for sequence analysis.  
2. **Bidirectional LSTM Network**:  
   - **Bidirectional LSTM layers**: Capture both forward and backward temporal dependencies in EEG sequences.  
   - **Dropout layers**: Prevent overfitting.  
   - **Batch Normalization layers**: Enhance training stability.  
3. **Output**: Dense layers with ReLU activations followed by a softmax layer for classification into AD, FTD, or CN categories.  

---

### ⚙️ Training Process  
- **Optimizer**: Adam  
- **Loss Function**: Categorical Cross-Entropy  
- **Class Weights**: Applied to address class imbalance  
- **Callbacks**:  
  - Early stopping  
  - Reduce learning rate on plateau  

### 🎯 Training Performance  
- **Test Accuracy**: 98%  

---

## 📋 Feature Engineering  

The features used for classification include:  
1. **Band Power**: Extracted from the following frequency bands:  
   - Delta  
   - Theta  
   - Alpha  
   - Beta  
   - Gamma  
2. **Spectral Entropy**: Quantifies the randomness and complexity of EEG signals.  

---

## 💾 Model Saving  

- **Model file**: `eegmodel.h5`  
- **Encoder file**: `encoder_model.h5`  

These models are used for both evaluation and real-time classification applications.  

---

## 📁 Project Structure  

```plaintext
EEG-Classification/
├── data/                    # Preprocessed EEG data
├── models/                  # Saved model files
│   ├── eegmodel.h5
│   ├── encoder_model.h5
├── notebooks/               # Jupyter notebooks for EDA and training
├── preprocessing/           # Preprocessing scripts (ICA, ASR, etc.)
├── results/                 # Visualizations and performance metrics
├── README.md                # Documentation

```
### 🚀 Installation

Follow the steps below to set up the project:

1. **Clone the Repository**  
   Clone this repository to your local machine:
   ```bash
   git clone https://github.com/your-username/eeg-classification.git
   cd eeg-classification
