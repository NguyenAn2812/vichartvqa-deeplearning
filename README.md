# 🇻🇳 Vietnamese Emotion Recognition — UIT-VSMEC

> Fine-tuned **PhoBERT-base-v2** for 7-class emotion classification on Vietnamese social media text.

**Task 2 — Deep Learning Final Project**  
Trường Đại học Tôn Đức Thắng — Khoa Công nghệ Thông tin

| Thành viên | MSSV |
|---|---|
| Nguyễn Trần Nhật An | 523H0115 |
| Chung Quang Vũ | 523H0196 |
| Nguyễn Nhật Chiều | 522H0133 |

---

## 📌 Tổng quan

Dự án fine-tune mô hình **PhoBERT-base-v2** trên bộ dữ liệu **UIT-VSMEC** để phân loại cảm xúc trong văn bản mạng xã hội tiếng Việt thành 7 nhãn:

| Nhãn | Cảm xúc |
|---|---|
| 😊 Enjoyment | Vui vẻ |
| 😢 Sadness | Buồn bã |
| 😡 Anger | Tức giận |
| 🤢 Disgust | Ghê tởm |
| 😨 Fear | Sợ hãi |
| 😲 Surprise | Ngạc nhiên |
| 😐 Other | Khác |

---

## 🗂️ Cấu trúc dự án

```
Vietnamese_Emotion_Recognition/
├── Vietnamese_Emotion_Recognition.ipynb   # Notebook chính
├── checkpoints/
│   └── best_model.pt                      # Checkpoint tốt nhất (sinh ra sau training)
├── saved_model/                           # Model + tokenizer đã lưu (sinh ra sau training)
│   ├── config.json
│   ├── tokenizer files...
│   ├── classifier_head.pt
│   └── emotion_config.json
├── label_dist.png                         # Biểu đồ phân phối nhãn
├── len_dist.png                           # Biểu đồ độ dài câu
├── training_curves.png                    # Loss / Accuracy / F1 theo epoch
├── confusion_matrix.png                   # Confusion matrix trên tập test
└── README.md
```

---

## 🚀 Hướng dẫn chạy

### 1. Chạy trên Google Colab (khuyến nghị)

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/nhatcheu/Vietnamese_Emotion_Recognition/blob/main/Vietnamese_Emotion_Recognition.ipynb)



### 2. Cài đặt thủ công

```bash
pip install transformers datasets tokenizers accelerate scikit-learn \
            pandas numpy matplotlib seaborn huggingface_hub tqdm gradio
```

### 3. Luồng chạy notebook

```
Section 0  → Cài thư viện
Section 1  → Import & cấu hình
Section 2  → Load dataset UIT-VSMEC từ HuggingFace
Section 3  → EDA (phân phối nhãn, độ dài câu)
Section 4  → Tạo Dataset class & DataLoaders
Section 5  → Định nghĩa model PhoBERTClassifier
Section 5b → (TÙY CHỌN) Load checkpoint có sẵn từ HF Hub → bỏ qua Section 6
Section 6  → Training (10 epochs, AdamW + Linear Warmup)
Section 7  → Đánh giá trên tập test (F1-macro, Accuracy, Confusion Matrix)
Section 8  → Inference demo với câu mẫu
Section 9  → Lưu model & push lên HuggingFace Hub (Do đã push lên HuggingFace rồi -> người test vui lòng bỏ qua bước này (Section 9) và chạy tiếp Section 10)
Section 10 → Gradio interactive demo
```

---

## 🏗️ Kiến trúc mô hình

```
Input Text (Vietnamese)
       ↓
PhoBERT-base-v2 (Encoder)
       ↓
  [CLS] token embedding (768-dim)
       ↓
  Dropout(0.3) → Linear(768→384) → ReLU → Dropout(0.15) → Linear(384→7)
       ↓
  Softmax → 7-class Emotion Label
```

---

## ⚙️ Cấu hình huấn luyện

| Tham số | Giá trị |
|---|---|
| Pretrained model | `vinai/phobert-base-v2` |
| Max sequence length | 128 |
| Epochs | 10 |
| Batch size | 16 |
| Learning rate | 2e-5 |
| Warmup ratio | 0.1 |
| Weight decay | 0.01 |
| Optimizer | AdamW |
| Loss function | CrossEntropyLoss |

---

## 📊 Kết quả

| Metric      | Score  |
|-------------|--------|
| Accuracy    | 0.6566 |
| F1-macro    | 0.6480 |
| F1-weighted | 0.6594 |

> Model checkpoint được lưu tại HuggingFace Hub: [RudiChill/vismec-emotion](https://huggingface.co/RudiChill/vismec-emotion)

---

## 📦 Dataset

- **Tên:** UIT-VSMEC
- **Nguồn:** [tridm/UIT-VSMEC](https://huggingface.co/datasets/tridm/UIT-VSMEC) trên HuggingFace
- **Số mẫu:** ~6,927 câu tiếng Việt từ mạng xã hội
- **Phân chia:** Train / Validation / Test

---

## 📚 Tham khảo

- Nguyen, D. Q., et al. (2020). *PhoBERT: Pre-trained language models for Vietnamese.* EMNLP Findings.
- Ho, V. A., et al. (2020). *Emotion Recognition for Vietnamese Social Media Text.* UIT-VSMEC dataset.
- HuggingFace Transformers: https://huggingface.co/docs/transformers

---

## 📄 License

Dự án phục vụ mục đích học thuật — **Tôn Đức Thắng University, 2026**.
