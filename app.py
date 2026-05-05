# ============================================================
# ViChartVQA Local Demo Web
# Models:
# A1: LSTM Decoder checkpoint from Hugging Face
# A2: Transformer Decoder checkpoint from Hugging Face
# B1: Qwen2-VL Zero-Shot
# B2: Qwen2-VL LoRA Fine-tuned
#
# Run:
# streamlit run app.py
# ============================================================

import os
import gc
import re
from collections import Counter

import torch
import torch.nn as nn
import torch.nn.functional as F
import streamlit as st
from PIL import Image

from datasets import load_dataset
from huggingface_hub import hf_hub_download
from transformers import (
    AutoTokenizer,
    AutoModel,
    ViTImageProcessor,
    AutoProcessor,
    Qwen2VLForConditionalGeneration,
)

from peft import PeftModel
import os
os.environ["TRANSFORMERS_VERBOSITY"] = "error"
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

# ============================================================
# 1. CONFIG
# ============================================================

class Config:
    HF_A1_REPO = "Zenng2812/vichartvqa-a1-lstm"
    HF_A2_REPO = "Zenng2812/vichartvqa-a2-transformer"
    HF_B2_REPO = "Zenng2812/vichartvqa-b2-qwen2vl-lora"

    A1_FILENAME = "best_model_a1.pth"
    A2_FILENAME = "best_model_a2.pth"

    QWEN_BASE_ID = "Qwen/Qwen2-VL-2B-Instruct"

    DATASET_ID = "Zenng2812/vqa-vietnamese-charts"

    MODEL_NAME_TEXT = "vinai/phobert-base"
    MODEL_NAME_IMG = "google/vit-base-patch16-224-in21k"

    MAX_QUESTION_LEN = 64
    MAX_ANSWER_LEN_A = 24

    QWEN_MIN_PIXELS = 32 * 28 * 28
    QWEN_MAX_PIXELS = 128 * 28 * 28

    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ANS_VOCAB_SIZE = 0


cfg = Config()


SYSTEM_PROMPT = (
    "Bạn là chuyên gia phân tích biểu đồ. "
    "Quan sát biểu đồ và trả lời câu hỏi bằng tiếng Việt. "
    "Câu trả lời cần ngắn gọn, tập trung vào xu hướng, so sánh hoặc nhận xét chính. "
    "Không giải thích dài dòng, không liệt kê quá nhiều số liệu."
)


# ============================================================
# 2. A1 / A2 MODEL CLASSES
# ============================================================

class AnswerVocab:
    SPECIAL_TOKENS = {
        "<PAD>": 0,
        "<SOS>": 1,
        "<EOS>": 2,
        "<UNK>": 3,
    }

    def __init__(self, dataset, max_size=1000):
        self.word2idx = dict(self.SPECIAL_TOKENS)
        self.idx2word = {i: w for w, i in self.word2idx.items()}

        all_answers = dataset["train"]["answer"]
        words = " ".join([str(x) for x in all_answers]).lower().split()

        for word, _ in Counter(words).most_common(max_size):
            if word not in self.word2idx:
                idx = len(self.word2idx)
                self.word2idx[word] = idx
                self.idx2word[idx] = word

    def decode(self, ids):
        skip = {
            self.word2idx["<PAD>"],
            self.word2idx["<SOS>"],
            self.word2idx["<EOS>"],
        }

        words = []
        for idx in ids:
            idx = int(idx)
            if idx in skip:
                continue
            words.append(self.idx2word.get(idx, "<UNK>"))

        return " ".join(words).replace(" <UNK>", "").strip()

    @property
    def size(self):
        return len(self.word2idx)


class CoAttention(nn.Module):
    def __init__(self, embed_dim):
        super().__init__()
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, img_feats, text_feats):
        q = self.q_proj(text_feats)
        k = self.k_proj(img_feats)
        v = self.v_proj(img_feats)

        scale = q.size(-1) ** 0.5
        attn = torch.matmul(q, k.transpose(-2, -1)) / scale
        attn = F.softmax(attn, dim=-1)

        fused = torch.matmul(attn, v)
        return self.norm(fused + text_feats)


class ChartVQAModel(nn.Module):
    def __init__(self, cfg, num_chart_types, decoder_type="LSTM"):
        super().__init__()

        self.decoder_type = decoder_type

        self.text_encoder = AutoModel.from_pretrained(cfg.MODEL_NAME_TEXT)
        self.image_encoder = AutoModel.from_pretrained(cfg.MODEL_NAME_IMG)

        hidden_dim = self.text_encoder.config.hidden_size

        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, num_chart_types),
        )

        self.fusion = CoAttention(hidden_dim)
        self.type_emb = nn.Embedding(num_chart_types, hidden_dim)

        self.ans_emb = nn.Embedding(cfg.ANS_VOCAB_SIZE, hidden_dim)

        if decoder_type == "LSTM":
            self.decoder = nn.LSTM(
                input_size=hidden_dim,
                hidden_size=hidden_dim,
                batch_first=True,
            )
        else:
            layer = nn.TransformerDecoderLayer(
                d_model=hidden_dim,
                nhead=8,
                batch_first=True,
            )
            self.decoder = nn.TransformerDecoder(
                decoder_layer=layer,
                num_layers=3,
            )

        self.fc_out = nn.Linear(hidden_dim, cfg.ANS_VOCAB_SIZE)

    @staticmethod
    def make_causal_mask(size, device):
        mask = torch.triu(torch.ones(size, size, device=device), diagonal=1)
        return mask.masked_fill(mask == 1, float("-inf"))

    def forward(
        self,
        input_ids,
        attention_mask,
        pixel_values,
        chart_labels=None,
        ans_input=None,
    ):
        text_out = self.text_encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
        ).last_hidden_state

        img_out = self.image_encoder(
            pixel_values=pixel_values
        ).last_hidden_state

        img_cls = img_out[:, 0, :]
        logits_cls = self.classifier(img_cls)

        if chart_labels is None:
            type_ids = logits_cls.argmax(dim=-1)
        else:
            type_ids = chart_labels

        context_token = self.type_emb(type_ids).unsqueeze(1)
        text_with_ctx = torch.cat([context_token, text_out], dim=1)

        fused = self.fusion(img_out, text_with_ctx)

        if ans_input is None:
            return logits_cls, fused

        ans_emb = self.ans_emb(ans_input)

        if self.decoder_type == "LSTM":
            h0 = fused.mean(dim=1).unsqueeze(0)
            c0 = torch.zeros_like(h0)
            out, _ = self.decoder(ans_emb, (h0, c0))
        else:
            tgt_mask = self.make_causal_mask(ans_input.size(1), ans_input.device)
            out = self.decoder(
                tgt=ans_emb,
                memory=fused,
                tgt_mask=tgt_mask,
            )

        logits_ans = self.fc_out(out)
        return logits_cls, logits_ans


# ============================================================
# 3. HELPERS
# ============================================================
def infer_answer_vocab_size_from_state_dict(state_dict):
    """
    Lấy vocab size của answer decoder trực tiếp từ checkpoint.
    A1/A2 checkpoint có key ans_emb.weight với shape [vocab_size, hidden_dim].
    """
    possible_keys = [
        "ans_emb.weight",
        "module.ans_emb.weight",
        "model.ans_emb.weight",
    ]

    for key in possible_keys:
        if key in state_dict:
            return state_dict[key].shape[0]

    # Nếu đã clean key rồi thì thường là ans_emb.weight
    for key in state_dict.keys():
        if key.endswith("ans_emb.weight"):
            return state_dict[key].shape[0]

    raise KeyError(
        "Không tìm thấy ans_emb.weight trong checkpoint, "
        "không thể suy ra ANS_VOCAB_SIZE."
    )
def get_state_dict_from_checkpoint(ckpt):
    if not isinstance(ckpt, dict):
        return ckpt

    possible_keys = [
        "model_state_dict",
        "model_state",
        "state_dict",
        "model",
    ]

    for key in possible_keys:
        if key in ckpt and isinstance(ckpt[key], dict):
            return ckpt[key]

    return ckpt


def clean_state_dict_keys(state_dict):
    cleaned = {}

    for k, v in state_dict.items():
        new_k = k

        if new_k.startswith("module."):
            new_k = new_k[len("module."):]

        if new_k.startswith("model."):
            new_k = new_k[len("model."):]

        cleaned[new_k] = v

    return cleaned


def get_bnb_config():
    if not torch.cuda.is_available():
        return None

    try:
        from transformers import BitsAndBytesConfig

        return BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
        )
    except Exception:
        return None


# ============================================================
# 4. LOAD A1 / A2
# ============================================================

@st.cache_resource(show_spinner=False)
def load_common_a_resources():
    dataset = load_dataset(cfg.DATASET_ID)

    ans_vocab = AnswerVocab(dataset)
    cfg.ANS_VOCAB_SIZE = ans_vocab.size

    unique_chart_types = sorted(set(dataset["train"]["chart_type"]))
    num_chart_types = len(unique_chart_types)

    tokenizer = AutoTokenizer.from_pretrained(cfg.MODEL_NAME_TEXT)
    image_processor = ViTImageProcessor.from_pretrained(cfg.MODEL_NAME_IMG)

    return ans_vocab, num_chart_types, tokenizer, image_processor


@st.cache_resource(show_spinner=False)
def load_a1_model():
    ans_vocab, num_chart_types, tokenizer, image_processor = load_common_a_resources()

    ckpt_path = hf_hub_download(
        repo_id=cfg.HF_A1_REPO,
        filename=cfg.A1_FILENAME,
        repo_type="model",
    )

    ckpt = torch.load(ckpt_path, map_location="cpu")
    state_dict = clean_state_dict_keys(get_state_dict_from_checkpoint(ckpt))

    # Quan trọng: set vocab size theo checkpoint trước khi tạo model
    cfg.ANS_VOCAB_SIZE = infer_answer_vocab_size_from_state_dict(state_dict)

    model = ChartVQAModel(
        cfg=cfg,
        num_chart_types=num_chart_types,
        decoder_type="LSTM",
    )

    missing, unexpected = model.load_state_dict(state_dict, strict=False)

    model.to(cfg.DEVICE)
    model.eval()

    return model, ans_vocab, tokenizer, image_processor, missing, unexpected
@st.cache_resource(show_spinner=False)
def load_a2_model():
    ans_vocab, num_chart_types, tokenizer, image_processor = load_common_a_resources()

    ckpt_path = hf_hub_download(
        repo_id=cfg.HF_A2_REPO,
        filename=cfg.A2_FILENAME,
        repo_type="model",
    )

    ckpt = torch.load(ckpt_path, map_location="cpu")
    state_dict = clean_state_dict_keys(get_state_dict_from_checkpoint(ckpt))

    # Quan trọng: set vocab size theo checkpoint trước khi tạo model
    cfg.ANS_VOCAB_SIZE = infer_answer_vocab_size_from_state_dict(state_dict)

    model = ChartVQAModel(
        cfg=cfg,
        num_chart_types=num_chart_types,
        decoder_type="Transformer",
    )

    missing, unexpected = model.load_state_dict(state_dict, strict=False)

    model.to(cfg.DEVICE)
    model.eval()

    return model, ans_vocab, tokenizer, image_processor, missing, unexpected
# ============================================================
# 5. PREDICT A1 / A2
# ============================================================

def predict_a_model(
    model,
    image,
    question,
    ans_vocab,
    tokenizer,
    image_processor,
    max_len=24,
):
    image = image.convert("RGB")
    model.eval()

    with torch.no_grad():
        pixel_values = image_processor(
            images=image,
            return_tensors="pt",
        ).pixel_values.to(cfg.DEVICE)

        enc = tokenizer(
            question,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=cfg.MAX_QUESTION_LEN,
        )

        input_ids = enc["input_ids"].to(cfg.DEVICE)
        attention_mask = enc["attention_mask"].to(cfg.DEVICE)

        _, fused = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            pixel_values=pixel_values,
        )

        sos = ans_vocab.word2idx["<SOS>"]
        eos = ans_vocab.word2idx["<EOS>"]

        predicted = []

        if model.decoder_type == "LSTM":
            current = torch.tensor([[sos]], device=cfg.DEVICE)

            h = fused.mean(dim=1).unsqueeze(0)
            c = torch.zeros_like(h)

            for _ in range(max_len):
                emb = model.ans_emb(current)
                out, (h, c) = model.decoder(emb, (h, c))
                logits = model.fc_out(out[:, -1, :])
                token = int(logits.argmax(dim=-1).item())

                if token == eos:
                    break

                predicted.append(token)
                current = torch.tensor([[token]], device=cfg.DEVICE)

        else:
            seq = torch.tensor([[sos]], device=cfg.DEVICE)

            for _ in range(max_len):
                emb = model.ans_emb(seq)
                mask = ChartVQAModel.make_causal_mask(seq.size(1), cfg.DEVICE)

                out = model.decoder(
                    tgt=emb,
                    memory=fused,
                    tgt_mask=mask,
                )

                logits = model.fc_out(out[:, -1, :])
                token = int(logits.argmax(dim=-1).item())

                if token == eos:
                    break

                predicted.append(token)

                next_token = torch.tensor([[token]], device=cfg.DEVICE)
                seq = torch.cat([seq, next_token], dim=1)

    return ans_vocab.decode(predicted)


# ============================================================
# 6. LOAD B1 / B2
# ============================================================

@st.cache_resource(show_spinner=False)
def load_b1_model():
    processor = AutoProcessor.from_pretrained(
        cfg.QWEN_BASE_ID,
        trust_remote_code=True,
        min_pixels=cfg.QWEN_MIN_PIXELS,
        max_pixels=cfg.QWEN_MAX_PIXELS,
    )

    bnb_config = get_bnb_config()

    if bnb_config is not None:
        model = Qwen2VLForConditionalGeneration.from_pretrained(
            cfg.QWEN_BASE_ID,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True,
        )
    else:
        dtype = torch.float16 if torch.cuda.is_available() else torch.float32
        model = Qwen2VLForConditionalGeneration.from_pretrained(
            cfg.QWEN_BASE_ID,
            torch_dtype=dtype,
            device_map="auto" if torch.cuda.is_available() else None,
            trust_remote_code=True,
        )

        if not torch.cuda.is_available():
            model.to("cpu")

    model.eval()
    return model, processor


@st.cache_resource(show_spinner=False)
def load_b2_model():
    try:
        processor = AutoProcessor.from_pretrained(
            cfg.HF_B2_REPO,
            trust_remote_code=True,
            min_pixels=cfg.QWEN_MIN_PIXELS,
            max_pixels=cfg.QWEN_MAX_PIXELS,
        )
    except Exception:
        processor = AutoProcessor.from_pretrained(
            cfg.QWEN_BASE_ID,
            trust_remote_code=True,
            min_pixels=cfg.QWEN_MIN_PIXELS,
            max_pixels=cfg.QWEN_MAX_PIXELS,
        )

    bnb_config = get_bnb_config()

    if bnb_config is not None:
        base_model = Qwen2VLForConditionalGeneration.from_pretrained(
            cfg.QWEN_BASE_ID,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True,
        )
    else:
        dtype = torch.float16 if torch.cuda.is_available() else torch.float32
        base_model = Qwen2VLForConditionalGeneration.from_pretrained(
            cfg.QWEN_BASE_ID,
            torch_dtype=dtype,
            device_map="auto" if torch.cuda.is_available() else None,
            trust_remote_code=True,
        )

        if not torch.cuda.is_available():
            base_model.to("cpu")

    base_model.config.use_cache = True

    model = PeftModel.from_pretrained(
        base_model,
        cfg.HF_B2_REPO,
        is_trainable=False,
    )

    model.eval()
    return model, processor


# ============================================================
# 7. PREDICT B1 / B2
# ============================================================

def predict_qwen_model(
    model,
    processor,
    image,
    question,
    max_new_tokens=96,
):
    image = image.convert("RGB")

    messages = [
        {
            "role": "system",
            "content": SYSTEM_PROMPT,
        },
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": str(question)},
            ],
        },
    ]

    text = processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    inputs = processor(
        text=[text],
        images=[image],
        padding=True,
        return_tensors="pt",
    )

    device = next(model.parameters()).device

    inputs = {
        k: v.to(device) if hasattr(v, "to") else v
        for k, v in inputs.items()
    }

    model.eval()

    with torch.no_grad():
        generated_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            num_beams=1,
        )

    generated_trimmed = [
        output_ids[len(input_ids):]
        for input_ids, output_ids in zip(inputs["input_ids"], generated_ids)
    ]

    prediction = processor.batch_decode(
        generated_trimmed,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )[0]

    return prediction.strip()


# ============================================================
# 8. STREAMLIT UI
# ============================================================

st.set_page_config(
    page_title="ViChartVQA Demo",
    page_icon="📊",
    layout="wide",
)

st.title("📊 ViChartVQA Demo")

st.markdown(
    """
Upload hoặc kéo ảnh biểu đồ vào, nhập câu hỏi, chọn model và bấm **Generate Answer**.  
Ứng dụng sẽ tự tải model từ Hugging Face trong lần chạy đầu tiên.
"""
)

with st.sidebar:
    st.header("⚙️ Model Settings")

    model_choice = st.selectbox(
        "Chọn model",
        [
            "A1 - LSTM Decoder",
            "A2 - Transformer Decoder",
            "B1 - Qwen2-VL Zero-Shot",
            "B2 - Qwen2-VL Fine-tuned",
        ],
        index=3,
    )

    max_new_tokens = st.slider(
        "Max new tokens cho B1/B2",
        min_value=16,
        max_value=160,
        value=96,
        step=8,
    )

    max_len_a = st.slider(
        "Max answer length cho A1/A2",
        min_value=5,
        max_value=50,
        value=24,
        step=1,
    )

    st.divider()

    st.write("Device:", str(cfg.DEVICE))

    if st.button("Clear model cache"):
        st.cache_resource.clear()
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        st.success("Đã clear cache. Hãy chạy lại prediction.")


col_left, col_right = st.columns([1, 1])

with col_left:
    uploaded_file = st.file_uploader(
        "Kéo ảnh vào đây hoặc chọn file",
        type=["png", "jpg", "jpeg", "webp"],
    )

    question = st.text_area(
        "Nhập câu hỏi",
        value="Biểu đồ này thể hiện xu hướng gì?",
        height=120,
    )

    run_button = st.button("🚀 Generate Answer", type="primary")


with col_right:
    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption="Ảnh đầu vào", use_container_width=True)
    else:
        image = None
        st.info("Hãy upload ảnh biểu đồ để bắt đầu.")


if run_button:
    if image is None:
        st.warning("Bạn chưa upload ảnh.")
    elif not question.strip():
        st.warning("Bạn chưa nhập câu hỏi.")
    else:
        try:
            with st.spinner(f"Đang load và chạy {model_choice}..."):
                if model_choice.startswith("A1"):
                    model, ans_vocab, tokenizer, image_processor, missing, unexpected = load_a1_model()

                    answer = predict_a_model(
                        model=model,
                        image=image,
                        question=question,
                        ans_vocab=ans_vocab,
                        tokenizer=tokenizer,
                        image_processor=image_processor,
                        max_len=max_len_a,
                    )

                    debug_info = {
                        "missing_keys": len(missing),
                        "unexpected_keys": len(unexpected),
                    }

                elif model_choice.startswith("A2"):
                    model, ans_vocab, tokenizer, image_processor, missing, unexpected = load_a2_model()

                    answer = predict_a_model(
                        model=model,
                        image=image,
                        question=question,
                        ans_vocab=ans_vocab,
                        tokenizer=tokenizer,
                        image_processor=image_processor,
                        max_len=max_len_a,
                    )

                    debug_info = {
                        "missing_keys": len(missing),
                        "unexpected_keys": len(unexpected),
                    }

                elif model_choice.startswith("B1"):
                    model, processor = load_b1_model()

                    answer = predict_qwen_model(
                        model=model,
                        processor=processor,
                        image=image,
                        question=question,
                        max_new_tokens=max_new_tokens,
                    )

                    debug_info = {}

                else:
                    model, processor = load_b2_model()

                    answer = predict_qwen_model(
                        model=model,
                        processor=processor,
                        image=image,
                        question=question,
                        max_new_tokens=max_new_tokens,
                    )

                    debug_info = {}

            st.success("✅ Prediction completed")

            st.markdown("### Question")
            st.write(question)

            st.markdown("### Answer")
            st.write(answer)

            if debug_info:
                with st.expander("Debug info"):
                    st.json(debug_info)

        except Exception as e:
            st.error("Có lỗi khi chạy demo.")
            st.exception(e)