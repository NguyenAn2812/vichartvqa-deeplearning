import ast
import json
import re
import time
import threading
import requests
import config


class KeyManager:
    def __init__(self):
        with open("api.txt", "r") as f:
            self.keys = [line.strip() for line in f if line.strip()]
        self.current_idx = 0
        self.lock        = threading.Lock()
        self.status      = {key: True for key in self.keys}

    def get_key(self):
        with self.lock:
            for _ in range(len(self.keys)):
                key = self.keys[self.current_idx]
                self.current_idx = (self.current_idx + 1) % len(self.keys)
                if self.status[key]:
                    return key
            print("All API keys rate-limited. Waiting 30s...")
            time.sleep(30)
            for key in self.keys:
                self.status[key] = True
            return self.keys[0]

    def mark_limit(self, key):
        with self.lock:
            self.status[key] = False


km = KeyManager()


def safe_parse_json(raw):
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        try:
            return ast.literal_eval(raw)
        except:
            try:
                return json.loads(raw.replace("'", '"'))
            except:
                return None


def call_llm(prompt, is_json=False):
    max_retries = len(km.keys) * 2

    for attempt in range(max_retries):
        api_key  = km.get_key()
        messages = []
        if is_json:
            messages.append({"role": "system", "content": "You output JSON only. No explanation, no markdown."})
        messages.append({"role": "user", "content": prompt})

        try:
            response = requests.post(
                url="https://openrouter.ai/api/v1/chat/completions",
                headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
                json={
                    "model": config.MODEL_NAME,
                    "messages": messages,
                    "temperature": 0.7,
                    **({"response_format": {"type": "json_object"}} if is_json else {})
                },
                timeout=60
            )

            if response.status_code == 429:
                km.mark_limit(api_key)
                continue

            if response.status_code in [400, 401, 403]:
                km.mark_limit(api_key)
                break

            data = response.json()

            if "error" in data:
                err = data["error"].get("message", "")
                if "rate" in err.lower():
                    km.mark_limit(api_key)
                continue

            content = data["choices"][0]["message"]["content"].strip()
            content = re.sub(r'<think>.*?</think>', '', content, flags=re.DOTALL).strip()
            return content.replace("```json", "").replace("```", "").strip()

        except requests.exceptions.Timeout:
            time.sleep(2)
        except Exception:
            time.sleep(1)

    return None


def get_single_chart_config(forced_type="bar"):
    raw = call_llm(config.PROMPT_CHART_CONFIG.format(forced_type=forced_type), is_json=True)
    try:
        data = safe_parse_json(raw)
        return data if data else None
    except:
        return None


def get_questions(markdown_data):
    raw = call_llm(config.PROMPT_GEN_QUESTIONS.format(markdown_data=markdown_data), is_json=True)
    try:
        data = safe_parse_json(raw)
        if not data:
            return []
        if isinstance(data, list):
            questions = data
        elif isinstance(data, dict):
            questions = data.get('questions', [])
        else:
            return []
        return questions if len(questions) == 4 else []
    except:
        return []


def get_answers_batch(markdown_data, questions):
    qs_formatted = "\n".join([f"{i+1}. {q}" for i, q in enumerate(questions)])
    prompt = f"""
Bảng dữ liệu:
{markdown_data}

Trả lời {len(questions)} câu hỏi sau:
{qs_formatted}

YÊU CẦU:
- Trả lời bằng NHẬN XÉT hoặc MÔ TẢ XU HƯỚNG, không đọc số liệu cụ thể.
  Ví dụ đúng: "Tăng dần đều qua các giai đoạn", "Giảm mạnh sau đỉnh giữa kỳ"
  Ví dụ sai: "Đạt 85 triệu vào tháng 3", "Tăng thêm 20 đơn vị"
- Mỗi câu trả lời 5-12 từ, tự nhiên như người xem biểu đồ nhận xét.
- Trả về JSON: {{"answers": ["đáp án 1", "đáp án 2", "đáp án 3", "đáp án 4"]}}
- Không giải thích gì thêm.
"""
    raw = call_llm(prompt, is_json=True)
    try:
        data = safe_parse_json(raw)
        if not data:
            return []
        answers = data.get('answers', [])
        return answers if len(answers) == len(questions) else []
    except:
        return []