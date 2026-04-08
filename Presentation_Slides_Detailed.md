# สไลด์ Presentation ฉบับละเอียด
## ระบบรู้จำอารมณ์จากเสียงพูดแบบรวมหลายภาษาด้วยเทคนิคการเรียนรู้เชิงลึก

---
---

# SLIDE 1 — ปกหน้า

# ระบบรู้จำอารมณ์จากเสียงพูด
# แบบรวมหลายภาษา
### Multilingual Speech Emotion Recognition Using Deep Learning

| | |
|---|---|
| **นักศึกษา** | รหัส 66070131 |
| **สาขาวิชา** | วิทยาการคอมพิวเตอร์ |
| **ภาคการศึกษา** | 2 / 2567 |
| **Hardware** | NVIDIA GeForce RTX 3060 (12 GB VRAM) |
| **Framework** | TensorFlow 2.x / Keras, Librosa, scikit-learn |

---
---

# SLIDE 2 — สารบัญ

### หัวข้อการนำเสนอ (ประมาณ 20–25 นาที)

| # | หัวข้อ | เวลา |
|---|---|---|
| 1 | ที่มาและความสำคัญ | 2 นาที |
| 2 | วัตถุประสงค์และขอบเขต | 2 นาที |
| 3 | ทฤษฎีและ Dataset ที่ใช้ | 3 นาที |
| 4 | สถาปัตยกรรมระบบ: Data Pipeline | 3 นาที |
| 5 | สถาปัตยกรรมโมเดล CNN + Bi-LSTM | 3 นาที |
| 6 | ผลการทดลองและ Confusion Matrix | 3 นาที |
| 7 | ปัญหาหลัก: Prosody Mismatch | 3 นาที |
| 8 | แนวทางในอนาคต: Per-Language Model | 3 นาที |
| 9 | สรุปและ Q&A | 3 นาที |

---
---

# SLIDE 3 — ที่มาและความสำคัญ (1/2)

## Speech Emotion Recognition (SER) คืออะไร?

**ระบบที่ให้คอมพิวเตอร์จำแนกอารมณ์จากเสียงพูดโดยอัตโนมัติ**

```
เสียงพูดเข้ามา  →  ประมวลผล  →  ผลลัพธ์อารมณ์
  "ฉันโกรธมาก!"      [ระบบ]      → ANGRY (Confidence 87%)
```

### การประยุกต์ใช้งานจริง

| สาขา | การใช้งาน |
|---|---|
| **Call Center** | ตรวจจับลูกค้าที่โกรธ → ส่งต่อพนักงานทันที |
| **สุขภาพจิต** | ตรวจจับสัญญาณซึมเศร้าจากเสียงพูด |
| **รถยนต์** | In-car Emotion Detection ปรับเพลง/อุณหภูมิตามอารมณ์ผู้ขับ |
| **IoT** | อุปกรณ์สมาร์ทโฮมที่ตอบสนองต่ออารมณ์ผู้ใช้ |

---
---

# SLIDE 4 — ที่มาและความสำคัญ (2/2)

## แนวคิดเริ่มต้นของโครงงาน

### ปัญหาของ SER ในปัจจุบัน
> โมเดล SER ส่วนใหญ่รองรับ **ภาษาเดียว** เท่านั้น
> → ถ้าต้องการรองรับ 10 ภาษา ต้องสร้างโมเดลถึง 10 ตัว

### สมมติฐานของโครงงาน
> **ถ้ารวมข้อมูลเสียงหลายภาษาแล้วฝึกโมเดลเดียว**
> โมเดลจะเรียนรู้ "รูปแบบอารมณ์ที่เป็นสากล" (Language-Independent Emotion Features)
> และทำงานได้ดีกับทุกภาษาโดยไม่ต้องแยกโมเดล

### สิ่งที่ค้นพบระหว่างพัฒนา
> สมมติฐานดังกล่าว **มีข้อจำกัดพื้นฐาน** เพราะเสียงพูดแต่ละภาษา
> มีโครงสร้าง Prosody (น้ำเสียง จังหวะ พลังงาน) ที่แตกต่างกันมาก
> → โมเดลเดียวไม่สามารถ Generalize ข้ามภาษาได้อย่างมีประสิทธิภาพ

---
---

# SLIDE 5 — วัตถุประสงค์และขอบเขต

## วัตถุประสงค์ 4 ข้อ

1. **พัฒนา** ระบบ SER แบบ Multilingual ด้วยสถาปัตยกรรม CNN + Bidirectional LSTM
2. **ศึกษาและวิเคราะห์** ปัญหา Prosody Mismatch ที่เกิดจากการรวมข้อมูลหลายภาษา
3. **ประเมินประสิทธิภาพ** ด้วย Accuracy, Confusion Matrix, Classification Report
4. **เสนอแนวทาง Future Work** สำหรับการพัฒนาต่อ (Per-Language Model)

## ขอบเขตของโครงงาน

| รายการ | รายละเอียด |
|---|---|
| **ภาษา** | ภาษาอังกฤษ (RAVDESS) + ภาษาเกาหลี (Korean Dataset) |
| **อารมณ์ที่จำแนก** | 5 ประเภท: Angry / Happy / Sad / Neutral / Surprise |
| **Feature หลัก** | MFCC 40 coefficients |
| **Feature ทางเลือก** | MFCC 128 + Mel Spectrogram (256 มิติ) |
| **เป้าหมาย Accuracy** | ≥ 75% บน Test Set |

### สิ่งที่อยู่นอกขอบเขต
- ไม่รองรับ Real-time Streaming
- ไม่มีการวิเคราะห์เนื้อหาคำพูด (NLP)
- ไม่รวมภาษาไทย (ยังขาด Open Source Dataset ที่มีคุณภาพ)

---
---

# SLIDE 6 — ทฤษฎีอารมณ์พื้นฐาน

## ทำไมถึงเลือกอารมณ์ 5 ประเภท?

**อ้างอิง: ทฤษฎีของ Ekman (1992)** — อารมณ์พื้นฐาน 6 ประเภทที่เป็นสากลข้ามวัฒนธรรม

| อารมณ์ | ลักษณะเด่นทางเสียง | เหตุใดเลือก/ไม่เลือก |
|---|---|---|
| **Angry** ✅ | Pitch สูง, Energy สูง, พูดเร็ว | มีใน Dataset ทั้งสอง |
| **Happy** ✅ | Pitch สูง, จังหวะเร็ว, น้ำเสียงสดใส | มีใน Dataset ทั้งสอง |
| **Sad** ✅ | Pitch ต่ำ, พูดช้า, Energy ต่ำ | มีใน Dataset ทั้งสอง |
| **Neutral** ✅ | Pitch ปานกลาง, สม่ำเสมอ | มีใน Dataset ทั้งสอง |
| **Surprise** ✅ | Pitch ขึ้นสูงฉับพลัน | มีใน Dataset ทั้งสอง |
| Fear ❌ | — | Korean Dataset ไม่มีข้อมูล |
| Disgust ❌ | — | Korean Dataset ไม่มีข้อมูล |

### ข้อค้นพบสำคัญจากทฤษฎี Ekman
> แม้การแสดงออกทาง **สีหน้า** จะเป็นสากล (Universal)
> แต่การแสดงออกทาง **เสียงพูด** มีความแตกต่างระหว่างวัฒนธรรมอย่างมีนัยสำคัญ
> → เป็นสาเหตุพื้นฐานของปัญหา Prosody Mismatch ในโครงงานนี้

---
---

# SLIDE 7 — Dataset: RAVDESS (ภาษาอังกฤษ)

## RAVDESS — Ryerson Audio-Visual Database of Emotional Speech and Song

| รายการ | รายละเอียด |
|---|---|
| **ผู้จัดทำ** | Livingstone & Russo (2018), มหาวิทยาลัย Ryerson, แคนาดา |
| **ผู้แสดง** | นักแสดงมืออาชีพ **24 คน** (12 ชาย / 12 หญิง) สำเนียง North American English |
| **สภาพแวดล้อม** | บันทึกใน **Anechoic Chamber** — ควบคุมเสียงสะท้อนและ Background Noise อย่างสมบูรณ์ |
| **รูปแบบไฟล์** | WAV, Mono, 48 kHz (Downsample เป็น 22,050 Hz) |
| **อารมณ์ครอบคลุม** | 8 อารมณ์ (ใช้เพียง 4: Neutral, Happy, Sad, Angry) |
| **จำนวนไฟล์** | 1,440 ไฟล์ (Speech) / ~192 ไฟล์ต่ออารมณ์ที่เลือก |
| **คุณภาพ** | **สูงมาก** — Label ถูกต้อง 100%, ไม่มี Background Noise |

### ระบบ Naming ของไฟล์ RAVDESS
```
03-01-05-01-01-01-12.wav
         ↑
     Emotion Code
     01=neutral  03=happy  04=sad  05=angry

→ parts[2] = '05' → label = 'angry'
```

---
---

# SLIDE 8 — Dataset: Korean Voice Emotion Dataset (ภาษาเกาหลี)

## Korean Voice Emotion Dataset

| รายการ | รายละเอียด |
|---|---|
| **แหล่งที่มา** | Hugging Face Datasets Hub (Open Source, ไม่มีค่าใช้จ่าย) |
| **วิธีโหลด** | Streaming Mode — ไม่ต้องดาวน์โหลดทั้งหมดในคราวเดียว |
| **อารมณ์** | 5 ประเภท: Angry, Happy, Sad, Neutral, Surprise |
| **จำนวนไฟล์/อารมณ์** | ~400–600 ไฟล์ (มากกว่า RAVDESS ต่ออารมณ์) |
| **คุณภาพ** | **ปานกลาง** — มี Background Noise บางไฟล์ สภาพแวดล้อมไม่ได้ควบคุม |

### จำนวนไฟล์รวมในแต่ละอารมณ์ (ประมาณการ)

| อารมณ์ | RAVDESS (EN) | Korean (KO) | รวม | สัดส่วน |
|---|---|---|---|---|
| Angry | ~192 | ~400–600 | ~600–800 | ~21% |
| Happy | ~192 | ~400–600 | ~600–800 | ~21% |
| Sad | ~192 | ~400–600 | ~600–800 | ~21% |
| Neutral | ~192 | ~400–600 | ~600–800 | ~21% |
| Surprise | ~96 | ~400–600 | ~500–700 | ~16% |

> ⚠️ Korean Dataset มีไฟล์มากกว่า RAVDESS → โมเดลเกิด **Bias** ไปทางภาษาเกาหลีในเชิงปริมาณ
> แต่คุณภาพ RAVDESS ดีกว่ามาก → โมเดลเรียนรู้ Pattern ภาษาอังกฤษได้ดีกว่าจริงๆ

---
---

# SLIDE 9 — Feature Extraction: MFCC

## MFCC (Mel-Frequency Cepstral Coefficients)

**มาตรฐานสำหรับ Speech Processing ตั้งแต่ปี 1980 (Davis & Mermelstein, 1980)**
จำลองการรับรู้เสียงของหูมนุษย์ผ่าน Mel Scale

### 4 ขั้นตอนการคำนวณ

```
① Framing & Windowing
   แบ่งสัญญาณออกเป็น Frame ขนาด ~25 ms ทับซ้อนกัน 10 ms
   ใช้ Hamming Window ลด Spectral Leakage
         ↓
② Fast Fourier Transform (FFT)
   แปลงแต่ละ Frame จาก Time Domain → Frequency Domain
         ↓
③ Mel Filter Bank
   ผ่าน Triangular Filters บน Mel Scale
   สูตร: m = 2595 × log₁₀(1 + f/700)
         ↓
④ Log + DCT (Discrete Cosine Transform)
   De-correlate ค่าต่างๆ ให้ Feature แต่ละมิติเป็นอิสระจากกัน
         ↓
   MFCC Vector (40 หรือ 128 ค่า ต่อ 1 Time Step)
```

### Feature ที่ใช้ในโครงงาน

| Feature | Shape | VRAM | โมเดลที่ใช้ |
|---|---|---|---|
| MFCC 40 | (T, 40) | ต่ำ | Train_Universal_Super.py **(หลัก)** |
| MFCC 128 + Mel Spectrogram | (T, 256) | สูง | Train_Model_RTX3060.py |

---
---

# SLIDE 10 — ภาพรวม Data Pipeline

## กระบวนการเตรียมข้อมูลทั้งหมด

```
┌─────────────────────────────────────────────────────────────┐
│                    DATA PIPELINE                            │
│                                                             │
│  ① File Discovery                                           │
│     os.walk() → รวบรวมไฟล์เสียงทุกไฟล์ใน dataset/          │
│          ↓                                                  │
│  ② Label Detection (2 วิธีอัตโนมัติ)                         │
│     Korean: ตรวจจาก /folder/angry/file.wav                  │
│     RAVDESS: ตรวจจาก filename "03-01-05-..." → parts[2]     │
│          ↓                                                  │
│  ③ Audio Preprocessing                                      │
│     Load (22,050 Hz) → Trim Silence → Pad/Cut (3 วินาที)   │
│          ↓                                                  │
│  ④ Feature Extraction                                       │
│     librosa.feature.mfcc() → shape (130, 40)               │
│          ↓                                                  │
│  ⑤ FILE-LEVEL SPLIT ← จุดสำคัญ! ป้องกัน Data Leakage       │
│     Train 70% / Val 15% / Test 15%                         │
│          ↓                                                  │
│  ⑥ Data Augmentation (เฉพาะ Train เท่านั้น)                  │
│     Noise + Pitch Shift + Time Stretch → ข้อมูล 3×          │
│          ↓                                                  │
│  ⑦ StandardScaler (Fit บน Train เท่านั้น → บันทึก .pkl)      │
│          ↓                                                  │
│  ⑧ Model Training → CNN + Bi-LSTM                          │
└─────────────────────────────────────────────────────────────┘
```

---
---

# SLIDE 11 — การป้องกัน Data Leakage

## Data Leakage คืออะไร และทำไมถึงอันตราย?

> **Data Leakage** = ข้อมูล Test Set "รั่วไหล" เข้าสู่กระบวนการ Train
> → Accuracy ที่รายงานสูงเกินจริง → เมื่อใช้งานจริงประสิทธิภาพต่ำกว่าที่คาด

### เปรียบเทียบวิธีที่ผิด vs ถูก

| | ❌ วิธีผิด (มี Leakage) | ✅ วิธีถูก (ป้องกัน Leakage) |
|---|---|---|
| **ลำดับ Augmentation** | Augment ทั้งหมด → แล้วค่อย Split | **Split ก่อน** → แล้วค่อย Augment เฉพาะ Train |
| **Scaler** | Fit Scaler บนข้อมูลรวม Train+Val+Test | **Fit เฉพาะ Train** → Transform Val/Test ด้วยค่าเดิม |
| **ผลที่ตามมา** | Accuracy สูงเกินจริง 5–15% | Accuracy สะท้อนความเป็นจริง |

### ตัวอย่างการเกิด Leakage จาก Augmentation
```
ไฟล์ A.wav (original) → Augment → A.wav / A_noise.wav / A_pitch.wav

Split แบบผิด:
  Test Set ← A.wav (ต้นฉบับ)
  Train Set ← A_noise.wav / A_pitch.wav   ← โมเดลเห็นเนื้อหาเดียวกันแล้ว!

Split แบบถูก:
  กำหนด A.wav อยู่ใน Test → A_noise, A_pitch ต้องอยู่ใน Test ด้วย
  (แต่ใน Test เราไม่ใช้ Augmented version → ใช้ original เท่านั้น)
```

---
---

# SLIDE 12 — Data Augmentation

## เทคนิคสร้างข้อมูลเพิ่ม (เฉพาะ Train Set)

**เหตุผล:** เพิ่มความหลากหลายของข้อมูลโดยไม่ต้องเก็บข้อมูลใหม่ → โมเดล Generalize ดีขึ้น

| เทคนิค | วิธีการ | จุดประสงค์ |
|---|---|---|
| **Gaussian Noise** | เพิ่ม Random Noise ความเข้มต่ำ | ทนต่อ Background Noise ในสภาพแวดล้อมจริง |
| **Pitch Shifting** | ปรับ Pitch ±0.7 Semitones | ทนต่อความแตกต่างระดับเสียงระหว่างผู้พูด |
| **Time Stretching** | ยืด/หดเวลา rate=0.8 | ทนต่อความเร็วในการพูดที่แตกต่างกัน |

### ผลลัพธ์ของ Augmentation
```
1 ไฟล์ต้นฉบับ
    ├── A_noise.wav    (+ Gaussian Noise)
    ├── A_pitch.wav    (Pitch Shift ±0.7 Semitones)
    └── A_stretch.wav  (Time Stretch rate=0.8)

Train Set เพิ่มขึ้น 3 เท่า โดยไม่ต้องเก็บข้อมูลเพิ่ม
```

> ⚠️ Label ยังคงเดิม → A_noise ยังเป็นอารมณ์เดิมกับ A.wav
> Augmentation ดัดแปลงเฉพาะ "รูปร่างเสียง" ไม่ใช่ "ความหมายอารมณ์"

---
---

# SLIDE 13 — สถาปัตยกรรมโมเดล CNN + Bi-LSTM

## โครงสร้างโมเดลทั้งหมด (653,061 Parameters)

```
Input: MFCC Feature Sequence
Shape: (Batch, 130 Time Steps, 40 Features)
                    │
         ┌──────────▼──────────┐
         │    CNN Block 1      │  Conv1D(256, kernel=5) + BN + MaxPool + Drop(0.3)
         │  สกัด Local Pattern │  Output: (Batch, 65, 256)
         └──────────┬──────────┘
                    │
         ┌──────────▼──────────┐
         │    CNN Block 2      │  Conv1D(128, kernel=5) + BN + MaxPool + Drop(0.3)
         │  สกัด High Pattern  │  Output: (Batch, 32, 128)
         └──────────┬──────────┘
                    │
         ┌──────────▼──────────┐
         │   Bi-LSTM Layer 1   │  Bidirectional LSTM(128, return_seq=True) + Drop(0.3)
         │  จับ Context 2 ทิศ  │  Output: (Batch, 32, 256)
         └──────────┬──────────┘
                    │
         ┌──────────▼──────────┐
         │   Bi-LSTM Layer 2   │  Bidirectional LSTM(64) + Drop(0.3)
         │  สรุป ทั้ง Sequence  │  Output: (Batch, 128)
         └──────────┬──────────┘
                    │
         ┌──────────▼──────────┐
         │    Dense Layers     │  Dense(64, ReLU, L2) + Drop(0.3)
         │    จำแนกอารมณ์      │  Dense(5, Softmax)
         └──────────┬──────────┘
                    │
Output: Probability ของ 5 อารมณ์ (รวม = 1.0)
```

---
---

# SLIDE 14 — รายละเอียด Parameter แต่ละชั้น

## Parameter Count แยกตามชั้น

| Layer | Output Shape | Parameters | สัดส่วน | หน้าที่ |
|---|---|---|---|---|
| Conv1D(256, k=5) | (Batch, 130, 256) | 51,456 | 7.9% | สกัด Local Temporal Pattern |
| BatchNorm | (Batch, 130, 256) | 1,024 | 0.2% | ทำให้ Training เสถียร |
| MaxPool + Dropout | (Batch, 65, 256) | 0 | — | Compress + Regularize |
| Conv1D(128, k=5) | (Batch, 65, 128) | 163,968 | 25.1% | สกัด High-level Pattern |
| BatchNorm | (Batch, 65, 128) | 512 | 0.1% | ทำให้ Training เสถียร |
| MaxPool + Dropout | (Batch, 32, 128) | 0 | — | Compress + Regularize |
| **Bi-LSTM(128)** | (Batch, 32, 256) | **263,168** | **40.3%** | จับ Sequential Dependency |
| Dropout | (Batch, 32, 256) | 0 | — | Regularize |
| **Bi-LSTM(64)** | (Batch, 128) | **164,352** | **25.2%** | สรุป Context ทั้ง Sequence |
| Dense(64) | (Batch, 64) | 8,256 | 1.3% | Feature Transformation |
| Dense(5, Softmax) | (Batch, 5) | 325 | 0.1% | Output: 5 อารมณ์ |
| **รวม** | | **653,061** | **100%** | Trainable ทั้งหมด |

> Bi-LSTM ชั้นแรกใช้ Parameter **มากที่สุด 40.3%** เพราะ LSTM มี Gate Mechanism ซับซ้อน
> (Forget Gate + Input Gate + Output Gate + Cell State) × 2 ทิศทาง

---
---

# SLIDE 15 — ทำไมถึงเลือก CNN + Bi-LSTM?

## เหตุผลในการเลือกสถาปัตยกรรม Hybrid

**อ้างอิง: Zhao et al. (2019) — งานวิจัยแสดงว่า CNN + LSTM ทำงานได้ดีสำหรับ SER**

### บทบาทของแต่ละส่วน

| ส่วน | สกัด Pattern ประเภทใด | ตัวอย่าง |
|---|---|---|
| **CNN** | Local Temporal Pattern — การเปลี่ยนแปลงในช่วงสั้นๆ | การขึ้นลงของ Pitch ใน 5 Frame (~100 ms) |
| **Bi-LSTM** | Sequential Context — ความสัมพันธ์ระยะยาว | น้ำเสียงที่เริ่มต้นสูงแล้วค่อยๆ ลงตลอดประโยค |

### ทำไมต้อง **Bidirectional**?
```
Forward LSTM  →→→→→→→→→→→→→→→  (อ่านซ้ายไปขวา)
Backward LSTM ←←←←←←←←←←←←←←  (อ่านขวาไปซ้าย)
                    ↓ รวมกัน
         เข้าใจบริบททั้งก่อนและหลัง

เช่น: น้ำเสียงขาลงท้ายประโยค → ช่วยยืนยันว่าเป็นอารมณ์เศร้าหรือเป็นกลาง
      ซึ่งต้องอ่านจากทั้งสองทิศทางพร้อมกัน
```

---
---

# SLIDE 16 — Training Configuration

## การตั้งค่าการฝึกสอนโมเดล

| Parameter | ค่าที่ใช้ | เหตุผล |
|---|---|---|
| **Optimizer** | Adam (lr=0.001) | ปรับ Learning Rate อัตโนมัติ (Kingma & Ba, 2014) |
| **Loss Function** | Categorical Cross-Entropy | มาตรฐาน Multi-class Classification |
| **Batch Size** | 64 / 32 / 16 | ขึ้นกับ VRAM ที่มีอยู่ |
| **Max Epochs** | 100–150 | หยุดเองด้วย EarlyStopping |
| **EarlyStopping** | patience=10 | ป้องกัน Overfitting |
| **ReduceLROnPlateau** | factor=0.5, patience=4–5 | Fine-tune ใกล้ Convergence |
| **Dropout** | 0.3 (30%) | ลด Overfitting ทุกชั้น |
| **L2 Regularization** | λ=0.001 | จำกัดขนาด Weight |
| **Mixed Precision** | float16 | เพิ่มความเร็ว 30–50% บน RTX 3060 Tensor Cores |

## โมเดลที่พัฒนาทั้งหมด 4 Version

| ไฟล์ | Feature | Batch | จุดเด่น |
|---|---|---|---|
| **Train_Universal_Super.py** | MFCC 40 | 64 | เร็ว เบา **โมเดลหลัก** |
| **Train_Test.py** | MFCC 40 | 32 | Anti-Leakage เข้มงวดสูงสุด |
| **Train_Model_RTX3060.py** | MFCC 128 + Mel | 32 | High-Resolution + Mixed Precision |
| **Train_model_res.py** | MFCC 128 + Mel | 16 | High-Resolution สำหรับ VRAM ต่ำ |

---
---

# SLIDE 17 — ผลการ Training

## Training Progress ของโมเดลหลัก (Train_Universal_Super.py)

| Epoch | Train Loss | Train Acc | Val Loss | Val Acc | สถานะ |
|---|---|---|---|---|---|
| 1 | 1.4821 | 30.12% | 1.3945 | 35.21% | เริ่มเรียนรู้ |
| 15 | 0.8234 | 68.21% | 0.9105 | 62.34% | กำลัง Improve |
| 35 | 0.5123 | 80.12% | 0.8932 | 71.45% | **Best Zone** |
| 47 | 0.4821 | 85.23% | 0.9456 | 71.98% | EarlyStopping หยุด |
| **Best (Ep.37)** | — | — | — | **~72%** | **Weights ที่ใช้จริง** |

### การวิเคราะห์ Curve
```
Training Loss    ↘↘↘↘↘↘↘↘↘↘↘↘↘↘  (ลดต่อเนื่อง)
Validation Loss  ↘↘↘↘↘↗↗↗↗↗↗↗↗↗  (ลดแล้วขึ้นที่ Ep.~38)
                           ↑
                     Overfitting เริ่ม

EarlyStopping จะ Restore Weights จาก Best Epoch (Val Loss ต่ำสุด)
แล้วหยุดเมื่อ Val Loss ไม่ดีขึ้นติดต่อกัน 10 Epoch
```

---
---

# SLIDE 18 — ผลบน Test Set

## Classification Report (Test Set — 1,548 ตัวอย่าง)

| อารมณ์ | Precision | Recall | F1-Score | Support | วิเคราะห์ |
|---|---|---|---|---|---|
| **Angry** | 0.72 | 0.72 | 0.72 | 320 | Energy สูงโดดเด่น → จำแนกได้ดี |
| **Happy** | 0.65 | 0.65 | 0.65 | 315 | Pitch Pattern ต่างกันมากระหว่างภาษา |
| **Sad** | **0.74** | **0.74** | **0.74** | 298 | Pitch ต่ำสม่ำเสมอในทุกภาษา → ดีที่สุด |
| **Neutral** | 0.71 | 0.71 | 0.71 | 310 | สับสนกับ Sad บ้าง |
| **Surprise** | 0.63 | 0.63 | 0.63 | 305 | สับสน Happy สูงสุด → แย่ที่สุด |
| **Macro Avg** | 0.69 | 0.69 | 0.69 | 1,548 | |
| **Overall Accuracy** | | | **68.21%** | 1,548 | ต่ำกว่าเป้าหมาย 75% |

### สรุปผล 3 ตัวชี้วัดหลัก

| Metric | เป้าหมาย | ได้จริง | ผล |
|---|---|---|---|
| Train Accuracy | ≥ 80% | **85.23%** | ✅ ผ่าน |
| Val Accuracy | ≥ 75% | **~72%** | ❌ ไม่ผ่าน |
| **Test Accuracy** | **≥ 75%** | **68.21%** | ❌ **ไม่บรรลุ** |

---
---

# SLIDE 19 — Confusion Matrix

## ความสับสนระหว่างอารมณ์แต่ละคู่

| Actual ↓ / Predicted → | Angry | Happy | Sad | Neutral | Surprise |
|---|---|---|---|---|---|
| **Angry** | **72%** | 5% | 8% | 10% | 5% |
| **Happy** | 8% | **65%** | 3% | 9% | **15%** |
| **Sad** | 5% | 3% | **74%** | **15%** | 3% |
| **Neutral** | 7% | 6% | **12%** | **71%** | 4% |
| **Surprise** | 10% | **18%** | 4% | 5% | **63%** |

### Pattern การสับสนที่สำคัญ

| คู่ที่สับสน | อัตราสับสน | สาเหตุ |
|---|---|---|
| **Surprise → Happy** | 18% | ทั้งคู่มี Pitch สูง + Energy สูง แต่รูปแบบต่างกันระหว่างภาษา |
| **Happy → Surprise** | 15% | Feature เหมือนกัน โมเดลแยกไม่ออก |
| **Sad → Neutral** | 15% | ทั้งคู่มี Pitch ต่ำ แต่ Duration Pattern ต่างกันตามภาษา |
| **Neutral → Sad** | 12% | Pitch ต่ำใน KO อาจถูกตีความว่าเศร้า |

> ✅ **Angry** ทำงานได้ดีที่สุด (72%) เพราะ Energy สูงเป็น Feature ที่โดดเด่นในทุกภาษา
> ❌ **Surprise** แย่ที่สุด (63%) เพราะมีลักษณะคล้าย Happy แต่ต่างภาษา

---
---

# SLIDE 20 — ปัญหาหลัก: Prosody Mismatch (1/2)

## Prosody คืออะไร?

**Prosody = คุณสมบัติเหนือระดับเสียงของภาษา** ได้แก่

| คุณสมบัติ | คำอธิบาย |
|---|---|
| **Pitch (F0)** | ความถี่พื้นฐาน — รับรู้เป็น "เสียงสูง-ต่ำ" |
| **Duration** | ความยาวของแต่ละพยางค์ |
| **Energy** | ระดับความดัง/พลังงานของเสียง |
| **Rhythm** | จังหวะและรูปแบบการพูดโดยรวม |

## ปัญหาของ Multilingual Model

```
MFCC ที่โมเดลเห็น = ลักษณะอารมณ์  +  ลักษณะภาษา  +  ลักษณะผู้พูด
                      (ต้องการ)        (Noise)         (Noise)

โมเดลไม่มีกลไกในการแยก 3 ส่วนนี้ออกจากกัน!
```

## ความแตกต่าง Prosody ระหว่าง EN และ KO

| คุณสมบัติ | ภาษาอังกฤษ | ภาษาเกาหลี |
|---|---|---|
| **Pitch Range** | กว้าง 100–400 Hz ขึ้นลงชัดเจน | แคบกว่า 100–280 Hz ค่อยเป็นค่อยไป |
| **Rhythm** | Stress-timed (ไม่สม่ำเสมอ) | Syllable-timed (สม่ำเสมอกว่า) |
| **Energy** | พุ่งสูงฉับพลันเมื่อโกรธ | เพิ่มขึ้นทีละน้อยแม้โกรธ |
| **Intonation** | ขึ้นลงมากเพื่อสื่อความหมาย | ขึ้นลงน้อยกว่า แต่เปลี่ยนความหมายคำ |

---
---

# SLIDE 21 — ปัญหาหลัก: Prosody Mismatch (2/2)

## ตัวอย่างเฉพาะ: "อารมณ์โกรธ" ในสองภาษา

| Feature | ภาษาอังกฤษ (Angry) | ภาษาเกาหลี (Angry) | ปัญหา |
|---|---|---|---|
| **Pitch** | สูงฉับพลัน 300–400 Hz | ค่อยๆ สูง 200–280 Hz | KO-Angry อาจถูกตีความว่าไม่ใช่ Angry |
| **Energy** | พุ่งสูงชัดเจน | เพิ่มทีละน้อย | EN ดัง, KO เบากว่า แม้ Angry เหมือนกัน |
| **Rhythm** | Stress-timed ไม่สม่ำเสมอ | Syllable-timed สม่ำเสมอ | Temporal Pattern ต่างกันสิ้นเชิง |

## ผลที่ตามมา 3 ประการ

**1. Confusion Between Language and Emotion**
> โมเดลไม่รู้ว่า Pitch สูงเกิดจาก "โกรธ" หรือ "ธรรมชาติของภาษาอังกฤษ"

**2. Overfitting Toward Majority Language**
> RAVDESS มีคุณภาพสูงและมีสัดส่วนดี → โมเดลเรียน Pattern EN เป็นหลัก
> เมื่อเจอเสียง KO → ใช้ Rule ของ EN ทำนาย → ผิด

**3. Poor Generalization Across Languages**
> Train Accuracy 85% แต่ Test Accuracy 68% → Gap 17%
> ภาษาอังกฤษ: ~80–82% | ภาษาเกาหลี: ~50–55% → เฉลี่ยได้ ~68%

---
---

# SLIDE 22 — สรุปข้อจำกัดทั้งหมด

## 5 สาเหตุที่โครงงานไม่บรรลุเป้าหมาย

| # | ข้อจำกัด | สาเหตุหลัก | ผลกระทบ |
|---|---|---|---|
| 1 | **Prosody Mismatch** | EN และ KO มีโครงสร้าง Pitch/Rhythm/Energy ต่างกันพื้นฐาน | โมเดลสับสนระหว่างลักษณะภาษากับลักษณะอารมณ์ |
| 2 | **Data Imbalance** | RAVDESS มีคุณภาพดีกว่า Korean Dataset มาก | โมเดล Bias ไปทางภาษาอังกฤษ |
| 3 | **Feature Entanglement** | MFCC รวม Emotion + Language + Speaker Features | ไม่มีกลไกแยก "สัญญาณอารมณ์" ออกจาก "สัญญาณภาษา" |
| 4 | **Cultural Difference** | การแสดงอารมณ์วัฒนธรรมตะวันออก ≠ ตะวันตก | ระดับความเข้มข้นต่างกัน EN แสดงรุนแรงกว่า KO |
| 5 | **Single Scaler Problem** | Fit Scaler บนข้อมูล Mixed Language | สถิติ Scale ไม่เหมาะกับภาษาใดภาษาหนึ่งโดยเฉพาะ |

### บทวิเคราะห์ระดับ Root Cause
```
ปัญหาทั้งหมดมาจากสมมติฐานพื้นฐานที่ผิด:
"อารมณ์เดียวกัน → MFCC เหมือนกันในทุกภาษา"

ความเป็นจริง:
"อารมณ์เดียวกัน → MFCC ต่างกันตามภาษา เพราะ Prosody ต่างกัน"

แก้ได้แค่ระดับ Architecture หรือ Feature Extraction เท่านั้น
ไม่ใช่การเพิ่มข้อมูลหรือ Fine-tune Hyperparameter
```

---
---

# SLIDE 23 — แนวทางในอนาคต: Per-Language Model

## แนวทาง Future Work ที่เสนอ

**แทนที่จะฝึกโมเดลเดียวสำหรับทุกภาษา → แยกโมเดลตามภาษา**

```
                ┌─────────────────────┐
เสียงพูดเข้ามา  │  Language Identifier │  ← ระบุภาษาอัตโนมัติ
                │  (Target: ≥ 95%)    │
                └─────────┬───────────┘
                          │
              ┌───────────┼───────────┐
              │           │           │
              ▼           ▼           ▼
        ┌──────────┐ ┌──────────┐ ┌──────────┐
        │ EN Model │ │ KO Model │ │ TH Model │
        │ CNN+LSTM │ │ CNN+LSTM │ │ CNN+LSTM │
        │  ≥ 85%   │ │  ≥ 80%   │ │  ≥ 75%   │
        └────┬─────┘ └────┬─────┘ └────┬─────┘
              └───────────┼───────────┘
                          │
                          ▼
              อารมณ์ + Confidence Score + ภาษาที่ตรวจพบ
```

### เปรียบเทียบ Unified Model vs Per-Language Model

| | Unified (ปัจจุบัน) | Per-Language (อนาคต) |
|---|---|---|
| **Accuracy** | ~68% (ผสมทุกภาษา) | คาดว่า ~85% ต่อภาษา |
| **Prosody** | ผสมกัน → สับสน | แยกภาษา → เรียนรู้ถูกต้อง |
| **การขยาย** | Retrain ทั้งหมดเมื่อเพิ่มภาษา | เพิ่มโมเดลใหม่ ไม่กระทบเดิม |
| **ความซับซ้อน** | น้อย (โมเดลเดียว) | มากขึ้น (หลายโมเดล + Router) |

---
---

# SLIDE 24 — Feature ที่ควรเปลี่ยน: Wav2Vec 2.0 / HuBERT

## แนวทางแก้ที่ Feature Level

**ปัญหา:** MFCC ไม่สามารถแยก Language Features ออกจาก Emotion Features ได้

**แนวทาง:** ใช้ Pre-trained Speech Representation Model

```
MFCC          → สะท้อน Acoustic Properties  (ได้รับอิทธิพลจากภาษาสูง)
Wav2Vec 2.0   → สะท้อน Semantic Speech Patterns (Language-neutral มากกว่า)
HuBERT        → Self-supervised Learning → เรียนรู้ Hidden Speech Units

EN-Angry MFCC  ≠  KO-Angry MFCC     (ปัญหาปัจจุบัน)
EN-Angry Wav2Vec ≈ KO-Angry Wav2Vec  (คาดว่าดีขึ้น)
```

### แนวทางอื่นเพิ่มเติม

| แนวทาง | วิธีการ | ข้อดี |
|---|---|---|
| **Adversarial Training** | บังคับโมเดลให้ทำนายอารมณ์ได้แต่ทำนายภาษาไม่ได้ | สร้าง Language-Invariant Representation |
| **Domain Adaptation** | Fine-tune โมเดล EN บนข้อมูล KO | ต้องการข้อมูล KO น้อยกว่า Train ใหม่ |
| **Multitask Learning** | Train ทั้ง Emotion + Language ID พร้อมกัน | บังคับโมเดลแยก Feature ได้เอง |

---
---

# SLIDE 25 — แผนพัฒนา 6 Phase

## แผนการพัฒนาในอนาคต

| Phase | งาน | เป้าหมาย | ความท้าทาย |
|---|---|---|---|
| **Phase 1** | รวบรวม Dataset เพิ่ม (KEMDy สำหรับ KO) | ≥ 5,000 ตัวอย่าง/อารมณ์/ภาษา | Dataset มี License จำกัด |
| **Phase 2** | Train EN Model แยก (RAVDESS only) | EN Accuracy ≥ 85% | — |
| **Phase 3** | Train KO Model แยก (Korean data only) | KO Accuracy ≥ 80% | ขาด Dataset คุณภาพ |
| **Phase 4** | พัฒนา Language Identifier | Lang ID ≥ 95% | ต้องการ Audio Language Dataset |
| **Phase 5** | รวม Router + Models เข้าด้วยกัน | End-to-end System | Latency เพิ่มขึ้น |
| **Phase 6** | ทดสอบ Real-world + ปรับปรุง | Overall Accuracy ≥ 80% | Edge Cases: Code-switching |

### ความท้าทายเพิ่มเติมที่ต้องแก้

| ความท้าทาย | รายละเอียด |
|---|---|
| **Dataset Scarcity** | ภาษาเกาหลีและไทยยังขาดแคลน Labeled Dataset ที่มีคุณภาพ |
| **Language ID Accuracy** | ถ้า Identifier ผิดพลาด → ส่งเสียงไปโมเดลผิดตัว → Accuracy ตก |
| **Latency** | หลายโมเดลใช้เวลาประมวลผลนานขึ้น → Trade-off กับ Accuracy |
| **Code-switching** | ผู้พูดสลับภาษากลางประโยค → ระบบยังไม่รองรับ |

---
---

# SLIDE 26 — สรุป

## สิ่งที่ทำสำเร็จในโครงงานนี้

### ✅ ด้าน Data Pipeline
- ระบบตรวจจับ Label อัตโนมัติ 2 วิธี (Path-based + Filename-based)
- ป้องกัน Data Leakage อย่างสมบูรณ์ (File-level Split ก่อนทุกกระบวนการ)
- Data Augmentation 3 เทคนิค เพิ่มข้อมูล Train 3 เท่า

### ✅ ด้านโมเดล
- สร้าง CNN + Bi-LSTM สำเร็จ 4 Version รองรับ Hardware ต่างระดับ
- Train Accuracy 85.23% — สถาปัตยกรรมเรียนรู้ข้อมูลได้ดี
- รวม 653,061 Trainable Parameters

### ✅ ด้านการวิเคราะห์
- วิเคราะห์ปัญหา Prosody Mismatch อย่างละเอียดเชิงลึก
- เสนอแนวทาง Future Work ที่ชัดเจนและปฏิบัติได้จริง

---

## สิ่งที่ยังไม่บรรลุ

| Metric | เป้าหมาย | ได้จริง | ส่วนต่าง |
|---|---|---|---|
| **Test Accuracy** | ≥ 75% | **68.21%** | **-6.79%** |

---

## บทเรียนสำคัญที่ได้

> **"การรวมข้อมูลหลายภาษาโดยไม่คำนึงถึง Prosody Mismatch**
> **ทำให้โมเดลไม่สามารถเรียนรู้อารมณ์ที่เป็นสากลได้"**
>
> สมมติฐาน Unified Multilingual Model มีข้อจำกัดที่ Feature Level
> ต้องแก้ที่ Architecture (Per-Language) หรือ Feature Extraction (Wav2Vec/HuBERT)

---
---

# SLIDE 27 — Q&A

# ขอบคุณ

## Key Takeaways

| | รายละเอียด |
|---|---|
| **โครงงาน** | Multilingual Speech Emotion Recognition |
| **ภาษา** | อังกฤษ (RAVDESS) + เกาหลี (Korean Dataset) |
| **อารมณ์** | 5 ประเภท: Angry / Happy / Sad / Neutral / Surprise |
| **สถาปัตยกรรม** | CNN + Bidirectional LSTM (653,061 Parameters) |
| **Feature** | MFCC 40 coefficients (หลัก) |
| **Hardware** | NVIDIA RTX 3060 (12 GB VRAM) |
| **Test Accuracy** | **68.21%** (เป้าหมาย 75%) |
| **ปัญหาหลัก** | **Prosody Mismatch** ระหว่าง EN และ KO |
| **แนวทางอนาคต** | Per-Language Model + Wav2Vec 2.0 / HuBERT |

---

## คำถามที่อาจถูกถาม (เตรียมคำตอบไว้)

**Q: ทำไมเลือกภาษาเกาหลี ไม่เลือกภาษาอื่น?**
→ เกาหลีมี Prosody ต่างจากอังกฤษมาก (Syllable-timed vs Stress-timed) ทำให้เห็นปัญหาชัดเจน + มี Open Dataset

**Q: ถ้า Train นานกว่านี้ Accuracy จะดีขึ้นไหม?**
→ ไม่มากนัก เพราะปัญหาอยู่ที่ Feature Distribution ต่างกัน ไม่ใช่เรื่อง Training epochs

**Q: ทำไม Sad ทำงานได้ดีที่สุด?**
→ Sad มี Pitch ต่ำ + Energy ต่ำ ซึ่งเป็น Pattern ที่คงเส้นคงวาในทุกภาษา แตกต่างจาก Happy/Surprise ที่ Pitch สูงแต่รูปแบบต่างกันมากระหว่างภาษา

**Q: Per-Language Model จะแก้ปัญหาได้จริงหรือ?**
→ ควรแก้ได้ เพราะแต่ละโมเดลเรียนรู้ Prosody Pattern ของภาษาตัวเองโดยเฉพาะ แต่ต้องอาศัย Language Identifier ที่แม่นยำ ≥ 95%

---
---

## หมายเหตุสำหรับผู้นำเสนอ

| สไลด์ | หัวข้อ | เวลา | จุดเน้น |
|---|---|---|---|
| 1–2 | ปก + สารบัญ | 1 นาที | แนะนำภาพรวม |
| 3–5 | ที่มา + วัตถุประสงค์ | 3 นาที | ปัญหาที่แก้ + ขอบเขต |
| 6–8 | ทฤษฎี + Dataset | 3 นาที | อธิบาย RAVDESS vs Korean |
| 9–12 | Data Pipeline + Anti-Leakage | 4 นาที | **เน้นการป้องกัน Leakage** |
| 13–16 | Model Architecture + Config | 4 นาที | **เน้น CNN+BiLSTM + เหตุผล** |
| 17–19 | ผลการทดลอง | 3 นาที | Training curve + Confusion Matrix |
| 20–22 | Prosody Mismatch | 4 นาที | **เน้นมาก — นี่คือ Discovery หลัก** |
| 23–25 | Future Work | 3 นาที | Per-Language + Wav2Vec |
| 26–27 | สรุป + Q&A | 3 นาที | Key Takeaways |
| **รวม** | | **~28 นาที** | |
