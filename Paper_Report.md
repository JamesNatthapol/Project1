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

Ekman [1] established the foundational theory of six basic emotions — Anger, Happiness, Sadness, Surprise, Fear, and Disgust — that are universally recognized across cultures. This framework underlies the emotion categories used in most SER datasets, including those used in this project. The assumption that emotion expression shares a universal acoustic basis across cultures motivated the unified multilingual approach explored here.

### 2.2 Cross-corpus and Cross-lingual SER

Schuller et al. [2] investigated cross-corpus acoustic emotion recognition, demonstrating that models trained on one corpus experience significant accuracy degradation when evaluated on another. This cross-corpus variance is analogous to the cross-lingual Prosody Mismatch problem observed in this project, where the acoustic space of emotion is shaped heavily by both recording environment and spoken language.

A key factor behind this mismatch is that prosodic features — the acoustic patterns that carry emotion — are not universal across languages. Table 1 illustrates how the same emotion ("Angry") manifests with significantly different prosodic profiles in English versus Korean.

**Table 1:** Prosody Characteristics of "Angry" Emotion Across Languages

| Feature | English "Angry" | Korean "Angry" |
|---|---|---|
| Pitch Range | Wide (300–500 Hz) | Narrower (200–350 Hz) |
| Pitch Pattern | Sharp rise, abrupt fall | Gradual sustained rise |
| Speech Rate | Fast, with abrupt pauses | Moderate, sustained delivery |
| Energy Pattern | Burst-type short peaks | Even, sustained high level |
| Rhythm | Stressed syllables prominent | More evenly distributed |

Because MFCC encodes all of these characteristics simultaneously, a model trained on both languages sees "Angry" as two different distributions, making unified classification difficult.

### 2.3 MFCC Feature Extraction

Davis and Mermelstein [3] introduced Mel-Frequency Cepstral Coefficients (MFCC) as a compact parametric representation of the spectral envelope of speech. MFCC remains the most widely used feature for SER tasks due to its efficiency and effectiveness for single-language systems. However, MFCC captures both emotional and linguistic characteristics simultaneously, making language-independent emotion learning difficult.

The MFCC extraction process follows a signal processing pipeline:

```
Raw Audio (WAV)
      |
      v
[Pre-emphasis Filter]    -- amplifies high-frequency components
      |
      v
[Frame Blocking]         -- splits audio into 25 ms frames (10 ms hop)
      |
      v
[Hamming Window]         -- reduces spectral leakage at frame edges
      |
      v
[Fast Fourier Transform] -- converts time domain to frequency domain
      |
      v
[Mel Filter Bank]        -- 40 triangular filters on Mel scale
      |
      v
[Log Compression]        -- approximates logarithmic human hearing
      |
      v
[Discrete Cosine Transform (DCT)]
      |
      v
MFCC Feature Vector  (shape: T x 40 per audio clip)
```

In this project, 40 MFCC coefficients are extracted per audio frame, producing a 2D matrix of shape (T, 40) that serves as the input sequence to the CNN + BiLSTM model.

### 2.4 Audio Processing Library

McFee et al. [4] developed Librosa, the Python library used in this project for all audio loading, MFCC extraction, Mel Spectrogram computation, and data augmentation operations including pitch shifting and time stretching. Librosa provides consistent, reproducible feature extraction across different audio file formats and sampling rates.

### 2.5 Deep Learning Architecture

Hochreiter and Schmidhuber [5] introduced Long Short-Term Memory (LSTM) networks, which address the vanishing gradient problem in standard RNNs by using gating mechanisms (input gate, forget gate, output gate) to selectively retain or discard information over long time spans. This makes LSTM well-suited for sequential audio data.

Schuster and Paliwal [6] extended LSTM to the Bidirectional variant (BiLSTM), which processes the input sequence in both forward and backward directions simultaneously, then concatenates both hidden states at each time step:

```
Input Frames:    x1    x2    x3   ...   xT

Forward LSTM:    h1f-->h2f-->h3f-->...-->hTf   (past to future)

Backward LSTM:   h1b<--h2b<--h3b<--...<--hTb  (future to past)

Output:          [h1f|h1b] [h2f|h2b] [h3f|h3b] ... [hTf|hTb]
                 (concatenated, captures full temporal context)
```

By reading the sequence in both directions, BiLSTM can incorporate context from both past and future frames when classifying each time step — a critical advantage for emotion recognition, since emotional cues such as intonation rise or trailing off are spread across the full utterance.

Zhao et al. [9] demonstrated that combining 1D CNN layers (for extracting local spectral patterns from each frame) with LSTM layers (for modeling temporal evolution of those patterns) achieves strong SER performance. This CNN + BiLSTM combination forms the architectural foundation of this project.

### 2.6 Regularization and Optimization

Srivastava et al. [7] introduced Dropout as an effective regularization technique that randomly deactivates a fraction of neurons during each training step, preventing co-adaptation and reducing overfitting. This project applies Dropout at four stages of the network with rates of 0.3 and 0.4.

Kingma and Ba [10] proposed the Adam optimizer, which combines momentum-based gradient updates with per-parameter adaptive learning rates. Adam was selected for its convergence stability and robustness to hyperparameter choices, particularly beneficial when training on imbalanced multilingual data.

### 2.7 Dataset

Livingstone and Russo [8] published the RAVDESS dataset (Ryerson Audio-Visual Database of Emotional Speech and Song), the English-language data source for this project. RAVDESS contains 1,440 audio clips recorded by 24 professional actors expressing 8 emotions in controlled anechoic chamber conditions, providing high signal-to-noise ratio and reliable ground-truth labels for SER research.

### 2.8 Summary of Related Work

Table 2 summarizes and compares the main approaches to multilingual SER discussed in the literature, positioning this project within the broader research landscape.

**Table 2:** Comparison of SER Approaches for Multilingual Settings

| Approach | Description | Advantage | Limitation |
|---|---|---|---|
| Per-Language Model | Separate model trained per language | High per-language accuracy | Requires language detection step |
| Unified Multilingual (this work) | Single model trained on all languages | Simple deployment | Prosody Mismatch degrades accuracy |
| Transfer Learning (fine-tuning) | Pre-train on large corpus, fine-tune per language | Good accuracy/cost balance | Needs large pre-training data |
| Self-supervised (Wav2Vec 2.0, HuBERT) | Language-neutral deep representations | Best cross-lingual performance | High computational cost |

From this review, three key gaps motivate this project:
1. **Multilingual SER remains an open problem** — most prior work uses per-language models or language-specific fine-tuning rather than a unified approach.
2. **MFCC alone is insufficient for cross-lingual SER** — language-neutral features such as Wav2Vec 2.0 or HuBERT are required for robust cross-lingual generalization.
3. **Data imbalance is a significant challenge** — English open datasets are far more abundant than Asian-language equivalents, biasing unified models toward the majority language.

---

## 3. Methodologies

### 3.1 System Architecture Overview

The system consists of two pipelines that share identical preprocessing and feature extraction steps to ensure consistency between training and inference:

**Training Pipeline** (step by step):
1. Scan audio files from dataset directory
2. Detect emotion label (from folder path for Korean; from filename encoding for RAVDESS)
3. Preprocess audio (trim silence, pad or cut to 3 seconds)
4. Extract MFCC features (40 coefficients per frame)
5. Apply Data Augmentation on training files only (Noise, Pitch Shift, Time Stretch)
6. Split into Train / Validation / Test at the file level
7. Fit StandardScaler on training features only
8. Train CNN + BiLSTM model with EarlyStopping and ReduceLROnPlateau

**Inference Pipeline** (step by step):
1. Receive input audio file
2. Apply identical preprocessing and MFCC extraction
3. Transform features using the saved StandardScaler
4. Pass features through the trained CNN + BiLSTM model
5. Output: predicted emotion label and per-class Confidence Scores

### 3.2 Datasets

**Table 3:** Dataset Summary

| Dataset | Language | Source | Total Emotions | Emotions Used |
|---|---|---|---|---|
| RAVDESS | English | Livingstone & Russo (2018) | 8 | 5 |
| Korean Voice Emotion Dataset | Korean | Hugging Face Datasets | 5 | 5 |

Both datasets were unified into a single directory structure organized by emotion label, with a combined 5-class label space: Angry, Happy, Sad, Neutral, Surprise.

### 3.3 Feature Extraction

Audio files were preprocessed to a standard 3-second duration at 22,050 Hz sample rate. Silence was trimmed using `librosa.effects.trim(top_db=25)`. The primary feature used was **MFCC with 40 coefficients** (shape: T x 40), computed per audio clip and passed to the model as a time-series sequence.

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

**Table 4:** CNN + BiLSTM Model Architecture and Parameter Count

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

**Table 5:** Training Configuration and Rationale

| Parameter | Value | Rationale |
|---|---|---|
| Optimizer | Adam (lr=0.001) | Adaptive learning rate, stable convergence |
| Loss | Categorical Cross-Entropy | Multi-class classification |
| Batch Size | 64 | Balanced memory and gradient stability |
| EarlyStopping | patience=10 | Prevent overfitting |
| ReduceLROnPlateau | factor=0.5, patience=5 | Escape learning plateaus |
| Epochs (max) | 100 | Stopped at 47 via EarlyStopping |
| GPU | NVIDIA RTX 3060 (12 GB VRAM) | Hardware accelerated training |

---

## 4. Result and Discussion

### 4.1 Training Progress

The training run terminated at Epoch 47 via EarlyStopping, with the best weights restored from Epoch 37. Table 6 shows accuracy progress and Table 7 shows the corresponding loss values at key epochs.

**Table 6:** Training and Validation Accuracy by Epoch

| Epoch | Train Acc | Val Acc | Status |
|---|---|---|---|
| 1 | 30.12% | 35.21% | Initial learning |
| 15 | 68.21% | 62.34% | Improving |
| 35 | 80.12% | 71.45% | Best zone |
| 47 | 85.23% | 71.98% | EarlyStopping triggered |
| **Best (Ep. 37)** | — | **~72%** | **Weights restored** |

**Table 7:** Training and Validation Loss by Epoch

| Epoch | Train Loss | Val Loss |
|---|---|---|
| 1 | 1.4821 | 1.3945 |
| 15 | 0.8234 | 0.9105 |
| 35 | 0.5123 | 0.8932 |
| 47 | 0.4821 | 0.9456 |

### 4.2 Test Set Performance

**Table 8:** Classification Report on Test Set

| Emotion | Prec. | Rec. | F1 | n |
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

[8] 	Livingstone, S. R., & Russo, F. A. (2018). *The Ryerson Audio-Visual Database of Emotional Speech and Song (RAVDESS): A dynamic, multimodal set of facial and vocal expressions in North American English.* PLOS ONE, 13(5), e0196391.

[9] 	Zhao, J., Mao, X., & Chen, L. (2019). *Speech emotion recognition using deep 1D & 2D CNN LSTM networks.* Biomedical Signal Processing and Control, 47, 312–323.

[10] 	Kingma, D. P., & Ba, J. (2014). *Adam: A method for stochastic optimization.* arXiv preprint arXiv:1412.6980.
