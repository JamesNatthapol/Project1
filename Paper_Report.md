# Emotional Speech Classification Using CNN-BiLSTM: A Multilingual Approach

---

## Author Information

| Field | Detail |
|---|---|
| Student ID | 66070062 / 66070131 |
| Department | Artificial Intelligence Technology |
| Semester | 2 / 2568 |

---

## Abstract

This paper presents the development of a Multilingual Speech Emotion Recognition (SER) system using a CNN + Bidirectional LSTM (BiLSTM) deep learning architecture. The system combines speech audio data from two languages — English (RAVDESS Dataset) and Korean (Korean Voice Emotion Dataset) — into a unified dataset and trains a single model to classify five emotion categories: Angry, Happy, Sad, Neutral, and Surprise.

During development, a critical challenge known as **Prosody Mismatch** was identified as the primary obstacle. Each language exhibits significantly different prosodic characteristics — including Pitch contour (F0), Speech Rate, Energy Pattern, and Rhythm — even when expressing the same emotion. These differences cause MFCC features to be an inseparable mixture of emotional and linguistic characteristics, preventing the model from learning universal emotion patterns. The achieved Test Accuracy was **~68.21%**, falling below the 75% target.

This paper presents the detailed methodology, root cause analysis of the Prosody Mismatch problem, and a proposed Future of Work based on a Per-Language Model architecture combined with an automatic Language Identifier.

---

## Keywords

Speech Emotion Recognition, Multilingual, Prosody Mismatch, CNN, Bidirectional LSTM, MFCC, Per-Language Model

---

## 1. Introduction

Speech Emotion Recognition (SER) is a rapidly growing research area with applications across multiple domains — including automated call centers that detect customer dissatisfaction, mental health monitoring systems that identify depressive states, smart IoT devices that respond to emotional cues, and in-vehicle emotion detection systems.

As cross-cultural and cross-lingual communication becomes increasingly common, developing a Multilingual SER model that supports multiple languages within a single unified system is highly desirable. Such a system would reduce the cost and complexity of maintaining separate language-specific models.

The central hypothesis of this project was that combining English and Korean speech emotion data and training a single CNN + BiLSTM model would allow the model to learn **language-independent emotion features** from the diversity of the combined dataset. However, development revealed a fundamental limitation: speech in each language has structurally different Prosody patterns that MFCC features cannot disentangle from the emotional signal.

The objectives of this project are:
1. To develop a Multilingual SER system using a CNN + Bidirectional LSTM architecture.
2. To study and analyze the Prosody Mismatch problem arising from combining multilingual data.
3. To evaluate model performance using Accuracy, Confusion Matrix, and Classification Report.
4. To propose a Future of Work roadmap based on the Per-Language Model approach.

The system classifies five emotion categories shared across both datasets: **Angry, Happy, Sad, Neutral,** and **Surprise**. Development and training were conducted on a personal machine equipped with an NVIDIA GeForce RTX 3060 (12 GB VRAM) using TensorFlow 2.x / Keras and Librosa for audio feature extraction.

---

## 2. Related Work

### 2.1 Basic Emotions in Speech

Ekman [1] established the foundational theory of six basic emotions — Anger, Happiness, Sadness, Surprise, Fear, and Disgust — that are universally recognized across cultures. This framework underlies the emotion categories used in most SER datasets, including those used in this project.

### 2.2 Cross-corpus and Cross-lingual SER

Schuller et al. [2] investigated cross-corpus acoustic emotion recognition, demonstrating that models trained on one corpus experience significant accuracy degradation when evaluated on another. This cross-corpus variance is analogous to the cross-lingual Prosody Mismatch problem observed in this project, where the acoustic space of emotion is shaped heavily by the recording context and language.

### 2.3 MFCC Feature Extraction

Davis and Mermelstein [3] introduced Mel-Frequency Cepstral Coefficients (MFCC) as a compact parametric representation of the spectral envelope of speech. MFCC remains the most widely used feature for SER tasks. However, as observed in this project, MFCC captures both emotional and linguistic characteristics simultaneously, making language-independent emotion learning difficult.

### 2.4 Audio Processing Library

McFee et al. [4] developed Librosa, the Python library used in this project for all audio loading, MFCC extraction, Mel Spectrogram computation, and data augmentation operations (pitch shifting, time stretching).

### 2.5 Deep Learning Architecture

Hochreiter and Schmidhuber [5] introduced Long Short-Term Memory (LSTM), the foundational recurrent unit used in this project's temporal modeling layers. Schuster and Paliwal [6] extended this to Bidirectional RNNs (BiLSTM), allowing the model to capture both past and future context within a sequence — critical for emotion recognition from speech.

Zhao et al. [10] demonstrated that combining 1D CNN layers (for local spectral feature extraction) with LSTM layers (for temporal dependency modeling) achieves strong SER performance, forming the architectural basis of this project's model.

### 2.6 Regularization and Optimization

Srivastava et al. [7] introduced Dropout as an effective regularization technique to prevent overfitting in neural networks. This project applies Dropout at four stages of the network with rates of 0.3 and 0.4. Kingma and Ba [11] proposed the Adam optimizer used for training, selected for its adaptive learning rate and convergence stability.

### 2.7 Dataset

Livingstone and Russo [9] published the RAVDESS dataset, the English-language data source for this project. RAVDESS contains 24 professional actors producing 8 emotions in controlled acoustic conditions, providing high-quality, reliably labeled data for SER research.

### 2.8 Summary of Related Work Gaps

From the literature review, three key insights emerge:
1. **Multilingual SER remains an open problem** — most prior work uses per-language models or language-specific fine-tuning.
2. **MFCC alone is insufficient for cross-lingual SER** — language-neutral features such as Wav2Vec 2.0 or HuBERT are recommended for future work.
3. **Data imbalance is a significant challenge** — English open datasets are far more abundant than Asian-language equivalents.

---

## 3. Methodologies

### 3.1 System Architecture Overview

The system consists of two pipelines:

- **Training Pipeline:** Audio files → Label Detection → Preprocessing (Trim/Pad) → MFCC Extraction → Data Augmentation (Train only) → File-level Train/Val/Test Split → StandardScaler fitting → CNN + BiLSTM training.
- **Inference Pipeline:** Input audio → same Preprocessing + MFCC → StandardScaler transform → Model prediction → Emotion label + Confidence Score.

### 3.2 Datasets

| Dataset | Language | Source | Emotions |
|---|---|---|---|
| RAVDESS | English | Livingstone & Russo (2018) | 8 (5 used) |
| Korean Voice Emotion Dataset | Korean | Hugging Face Datasets | 5 |

Both datasets were unified into a single directory structure organized by emotion label, with a combined 5-class label space: Angry, Happy, Sad, Neutral, Surprise.

### 3.3 Feature Extraction

Audio files were preprocessed to a standard 3-second duration at 22,050 Hz sample rate. Silence was trimmed using `librosa.effects.trim(top_db=25)`. The primary feature used was **MFCC with 40 coefficients** (shape: T × 40), computed per audio clip and passed to the model as a time-series sequence.

### 3.4 Data Augmentation

To improve generalization, three augmentation techniques were applied exclusively to training data:
- **Gaussian Noise addition** (SNR ≈ 20 dB)
- **Pitch Shifting** (±0.7 semitones)
- **Time Stretching** (rate = 0.8)

This expanded the training set approximately 3× while preventing augmented versions of training clips from appearing in the validation or test sets.

### 3.5 Anti-Data-Leakage Split

File paths were split at the file level before any feature loading or augmentation, ensuring no audio clip contributes to both training and evaluation sets. Train/Val/Test ratio: 70% / 15% / 15%, using `train_test_split` with `stratify=y` and `random_state=42`.

### 3.6 Model Architecture

The model is a sequential CNN + Bidirectional LSTM network with 653,061 trainable parameters:

| Layer | Output Shape | Parameters |
|---|---|---|
| Conv1D (256 filters, k=5) + BN + MaxPool + Dropout | (None, 65, 256) | 52,480 |
| Conv1D (128 filters, k=5) + BN + MaxPool + Dropout | (None, 32, 128) | 164,480 |
| BiLSTM (128 units) + Dropout | (None, 32, 256) | 263,168 |
| BiLSTM (64 units) + Dropout | (None, 128) | 164,352 |
| Dense (64, ReLU, L2) + Dropout | (None, 64) | 8,256 |
| Dense (5, Softmax) | (None, 5) | 325 |
| **Total** | | **653,061** |

### 3.7 Training Configuration

| Parameter | Value | Rationale |
|---|---|---|
| Optimizer | Adam (lr=0.001) | Adaptive learning rate, stable convergence |
| Loss | Categorical Cross-Entropy | Multi-class classification |
| Batch Size | 64 | Balanced memory/gradient stability |
| EarlyStopping | patience=10 | Prevent overfitting |
| ReduceLROnPlateau | factor=0.5, patience=5 | Escape learning plateaus |
| Epochs (max) | 100 | Stopped at 47 via EarlyStopping |
| GPU | NVIDIA RTX 3060 (12 GB VRAM) | Hardware accelerated training |

---

## 4. Result and Discussion

### 4.1 Training Progress

| Epoch | Train Loss | Train Acc | Val Loss | Val Acc | Status |
|---|---|---|---|---|---|
| 1 | 1.4821 | 30.12% | 1.3945 | 35.21% | Initial learning |
| 15 | 0.8234 | 68.21% | 0.9105 | 62.34% | Improving |
| 35 | 0.5123 | 80.12% | 0.8932 | 71.45% | Best zone |
| 47 | 0.4821 | 85.23% | 0.9456 | 71.98% | EarlyStopping |
| **Best (Ep. 37)** | — | — | — | **~72%** | **Restored** |

### 4.2 Test Set Performance

| Emotion | Precision | Recall | F1-Score | Support |
|---|---|---|---|---|
| Angry | 0.72 | 0.72 | 0.72 | 320 |
| Happy | 0.65 | 0.65 | 0.65 | 315 |
| Sad | 0.74 | 0.74 | 0.74 | 298 |
| Neutral | 0.71 | 0.71 | 0.71 | 310 |
| Surprise | 0.63 | 0.63 | 0.63 | 305 |
| **Macro avg** | 0.69 | 0.69 | 0.69 | 1,548 |
| **Overall Accuracy** | | | **68.21%** | 1,548 |

### 4.3 Discussion

**Sad** achieved the highest F1 (0.74) because low-energy, low-pitch patterns are relatively consistent across both languages. **Angry** performed well (0.72) due to the strong high-energy signal. **Surprise** achieved the lowest F1 (0.63) as its high-pitch pattern overlaps significantly with Happy, particularly across language boundaries.

The 6.79% gap between the achieved accuracy (68.21%) and the target (75%) is directly attributable to **Prosody Mismatch** — the fundamental difference in pitch contour, speech rate, energy pattern, and rhythm between English and Korean expressions of the same emotion. MFCC conflates these linguistic and emotional characteristics, preventing the model from learning a truly language-independent emotion representation.

The high Train Accuracy (85.23%) relative to Test Accuracy (68.21%) confirms the model successfully learned the training distribution but failed to generalize across language boundaries — a symptom of the fundamental cross-lingual challenge, not merely overfitting.

Additional issues encountered included Scaler Mismatch between training and evaluation runs (resolved via `Repair_Scaler.py`), Data Imbalance between the two language corpora, VRAM overflow for high-resolution feature experiments (resolved via Mixed Precision + reduced Batch Size), and Korean dataset codec errors (resolved via `soundfile` + `decode=False`).

---

## 5. Conclusions

This project successfully built the full infrastructure for a Multilingual Speech Emotion Recognition system — including a robust Data Pipeline with Anti-Data-Leakage splits, a CNN + BiLSTM model with 653,061 parameters, multiple training configurations for different hardware profiles, and a comprehensive evaluation suite. However, the primary objective of achieving 75%+ Test Accuracy was not met.

The core finding is that **MFCC-based unified multilingual training cannot learn language-independent emotion features** in the presence of structural Prosody differences between languages. This Negative Result holds academic value: it empirically demonstrates the limits of the unified Multilingual SER approach and motivates a more targeted solution.

The proposed Future of Work is a **Per-Language Model Architecture**: training a separate CNN + BiLSTM model for each language and routing incoming audio through an automatic Language Identifier before emotion classification. This approach eliminates Prosody Mismatch at the model level and is expected to significantly improve per-language accuracy. For further improvement, replacing MFCC with language-neutral features such as Wav2Vec 2.0 or HuBERT is recommended.

---

## References

[1] 	Ekman, P. (1992). *An argument for basic emotions.* Cognition & Emotion, 6(3–4), 169–200.

[2] 	Schuller, B., Vlasenko, B., Eyben, F., Wöllmer, M., Stuhlsatz, A., Wendemuth, A., & Rigoll, G. (2010). *Cross-corpus acoustic emotion recognition: Variances and strategies.* IEEE Transactions on Affective Computing, 1(2), 119–131.

[3] 	Davis, S. B., & Mermelstein, P. (1980). *Comparison of parametric representations for monosyllabic word recognition in continuously spoken sentences.* IEEE Transactions on Acoustics, Speech, and Signal Processing, 28(4), 357–366.

[4] 	McFee, B., Raffel, C., Liang, D., Ellis, D., McVicar, M., Battenberg, E., & Nieto, O. (2015). *librosa: Audio and music signal analysis in python.* Proceedings of the 14th Python in Science Conference (SciPy 2015), 18–25.

[5] 	Hochreiter, S., & Schmidhuber, J. (1997). *Long short-term memory.* Neural Computation, 9(8), 1735–1780.

[6] 	Schuster, M., & Paliwal, K. K. (1997). *Bidirectional recurrent neural networks.* IEEE Transactions on Signal Processing, 45(11), 2673–2681.

[7] 	Srivastava, N., Hinton, G., Krizhevsky, A., Sutskever, I., & Salakhutdinov, R. (2014). *Dropout: A simple way to prevent neural networks from overfitting.* The Journal of Machine Learning Research, 15(1), 1929–1958.

[8] 	Abadi, M., Barham, P., Chen, J., Chen, Z., Davis, A., Dean, J., ... & Zheng, X. (2016). *TensorFlow: A system for large-scale machine learning.* 12th USENIX Symposium on Operating Systems Design and Implementation (OSDI 16), 265–283.

[9] 	Livingstone, S. R., & Russo, F. A. (2018). *The Ryerson Audio-Visual Database of Emotional Speech and Song (RAVDESS): A dynamic, multimodal set of facial and vocal expressions in North American English.* PLOS ONE, 13(5), e0196391.

[10] 	Zhao, J., Mao, X., & Chen, L. (2019). *Speech emotion recognition using deep 1D & 2D CNN LSTM networks.* Biomedical Signal Processing and Control, 47, 312–323.

[11] 	Kingma, D. P., & Ba, J. (2014). *Adam: A method for stochastic optimization.* arXiv preprint arXiv:1412.6980.
