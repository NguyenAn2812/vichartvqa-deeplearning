import json
import os
import re
from tqdm import tqdm
import config


def normalize_text(text, is_question=False):
    if not isinstance(text, str) or not text:
        return text

    text = re.sub(r'\/+', ' ', text)
    text = re.sub(r'\?+', '?', text)
    text = text.strip().strip('/-. ')

    months = {
        'january': 'Tháng 1', 'february': 'Tháng 2', 'march': 'Tháng 3',
        'april': 'Tháng 4', 'may': 'Tháng 5', 'june': 'Tháng 6',
        'july': 'Tháng 7', 'august': 'Tháng 8', 'september': 'Tháng 9',
        'october': 'Tháng 10', 'november': 'Tháng 11', 'december': 'Tháng 12',
        'jan': 'Tháng 1', 'feb': 'Tháng 2', 'mar': 'Tháng 3',
        'apr': 'Tháng 4', 'jun': 'Tháng 6', 'jul': 'Tháng 7',
        'aug': 'Tháng 8', 'sep': 'Tháng 9', 'oct': 'Tháng 10',
        'nov': 'Tháng 11', 'dec': 'Tháng 12'
    }

    quarters = {'q1': 'Quý 1', 'q2': 'Quý 2', 'q3': 'Quý 3', 'q4': 'Quý 4'}

    others = {
        r'\byes\b': 'Có',
        r'\bno\b': 'Không',
        r'\btrue\b': 'Đúng',
        r'\bfalse\b': 'Sai',
        r'\bpercent\b': '%',
        r'\bbillion\b': 'tỷ',
        r'\bmillion\b': 'triệu',
        r'\baverage\b': 'trung bình',
        r'\btotal\b': 'tổng cộng',
        r'\bmaximum\b': 'cao nhất',
        r'\bminimum\b': 'thấp nhất',
        r'\bincrease\b': 'tăng',
        r'\bdecrease\b': 'giảm'
    }

    for eng, vie in months.items():
        text = re.sub(rf'\b{eng}\b', vie, text, flags=re.IGNORECASE)
    for eng, vie in quarters.items():
        text = re.sub(rf'\b{eng}\b', vie, text, flags=re.IGNORECASE)
    for pattern, repl in others.items():
        text = re.sub(pattern, repl, text, flags=re.IGNORECASE)

    punctuation_fixes = {
        r'\s+\?': '?',
        r'\s+\.': '.',
        r'\s+,': ',',
        r'(\d)\s+\/\s+(\d)': r'\1/\2',
    }
    for pattern, repl in punctuation_fixes.items():
        text = re.sub(pattern, repl, text)

    text = re.sub(r'\s+', ' ', text).strip()

    if is_question:
        if not text.endswith('?'):
            text += '?'
    else:
        text = text.rstrip('?')

    if text:
        text = text[0].upper() + text[1:]

    return text


def run_normalization():
    if not os.path.exists(config.DATASET_FILE):
        print("Dataset JSON not found.")
        return

    with open(config.DATASET_FILE, "r", encoding="utf-8") as f:
        dataset = json.load(f)

    for item in tqdm(dataset, desc="Normalizing"):
        item['question'] = normalize_text(item['question'], is_question=True)
        item['answer']   = normalize_text(item['answer'],   is_question=False)

    with open(config.DATASET_FILE, "w", encoding="utf-8") as f:
        json.dump(dataset, f, ensure_ascii=False, indent=4)

    print(f"Done. {len(dataset)} records normalized.")


if __name__ == "__main__":
    run_normalization()