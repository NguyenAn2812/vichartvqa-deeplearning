import os
import json
import re
import argparse
from collections import Counter, defaultdict
from pathlib import Path

DEFAULT_DATASET = "./vqa_flat_dataset/vqa_dataset.json"
DEFAULT_IMG_DIR = "./vqa_flat_dataset/images"
DEFAULT_MD_DIR  = "./vqa_flat_dataset/markdown"

SEP  = "=" * 60
SEP2 = "-" * 60


def word_count(text: str) -> int:
    return len(text.strip().split())

def char_count(text: str) -> int:
    return len(text.strip())

def avg(lst):
    return sum(lst) / len(lst) if lst else 0

def median(lst):
    if not lst: return 0
    s = sorted(lst)
    n = len(s)
    return s[n // 2] if n % 2 else (s[n//2 - 1] + s[n//2]) / 2

def pct(part, total):
    return f"{100 * part / total:.1f}%" if total else "0%"

def detect_question_direction(question: str) -> str:
    q = question.lower()
    if any(k in q for k in ["xu huong", "thay doi", "bien dong", "tang", "giam", "on dinh"]):
        return "Trend"
    if any(k in q for k in ["khi", "tuong quan", "moi quan he", "quan he", "so voi"]):
        return "Correlation"
    if any(k in q for k in ["cao nhat", "thap nhat", "dinh", "day", "dan dau", "noi bat", "giai doan nao", "muc nao"]):
        return "Extremum"
    if any(k in q for k in ["nua dau", "nua cuoi", "khac biet", "so sanh", "hon", "kem"]):
        return "Comparison"
    if any(k in q for k in ["bien dong nhieu", "bien dong it", "on dinh hay", "nhieu hay it"]):
        return "Volatility"
    return "Other"

def detect_answer_category(answer: str) -> str:
    a = answer.lower()
    if any(k in a for k in ["tang dan", "tang deu", "tang manh", "tang lien tuc"]):
        return "Upward"
    if any(k in a for k in ["giam dan", "giam deu", "giam manh", "giam lien tuc"]):
        return "Downward"
    if any(k in a for k in ["on dinh", "khong doi", "it bien dong"]):
        return "Stable"
    if any(k in a for k in ["bien dong", "khong on dinh", "dao dong", "len xuong"]):
        return "Volatile"
    if any(k in a for k in ["dat dinh", "cham day", "cao nhat", "thap nhat"]):
        return "Extremum"
    if any(k in a for k in ["vuot troi", "dan dau", "phan hoa", "bat kip"]):
        return "Comparative"
    return "Descriptive"


def analyze(dataset_path, img_dir, md_dir, export_latex=False):
    if not os.path.exists(dataset_path):
        print(f"Dataset not found: {dataset_path}")
        return
    with open(dataset_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list) or len(data) == 0:
        print("Dataset is empty or malformed.")
        return

    total_qa     = len(data)
    chart_ids    = list({item["id"] for item in data})
    total_charts = len(chart_ids)

    print(f"\n{SEP}")
    print("  ViChartVQA -- Dataset Analysis Report")
    print(SEP)

    print(f"\n{'[1] OVERVIEW':^60}")
    print(SEP2)
    print(f"  Total QA pairs           : {total_qa:,}")
    print(f"  Total unique charts      : {total_charts:,}")
    print(f"  Avg QA per chart         : {avg([total_qa / total_charts]):.2f}")
    qa_per_chart = Counter(item["id"] for item in data)
    perfect4 = sum(1 for v in qa_per_chart.values() if v == 4)
    print(f"  Charts with exactly 4 QA : {perfect4:,}  ({pct(perfect4, total_charts)})")

    print(f"\n{'[2] CHART TYPE DISTRIBUTION':^60}")
    print(SEP2)
    type_counter = Counter(item.get("type", "unknown") for item in data)
    chart_type_counter = Counter()
    for item in data:
        chart_type_counter[item["id"]] = item.get("type", "unknown")
    unique_type_dist = Counter(chart_type_counter.values())
    print(f"  {'Type':<12} {'QA pairs':>10} {'%':>8}  {'Charts':>8} {'%':>8}")
    print(f"  {'-'*52}")
    for t in ["bar", "line", "area", "pie"]:
        qa_n = type_counter.get(t, 0)
        ch_n = unique_type_dist.get(t, 0)
        print(f"  {t:<12} {qa_n:>10,} {pct(qa_n, total_qa):>8}  {ch_n:>8,} {pct(ch_n, total_charts):>8}")
    other_qa = sum(v for k, v in type_counter.items() if k not in ["bar", "line", "area", "pie"])
    if other_qa:
        print(f"  {'other':<12} {other_qa:>10,} {pct(other_qa, total_qa):>8}")

    print(f"\n{'[3] QUESTION ANALYSIS':^60}")
    print(SEP2)
    q_lens_word = [word_count(item["question"]) for item in data]
    q_lens_char = [char_count(item["question"]) for item in data]
    print(f"  Word length  -- min: {min(q_lens_word)}, max: {max(q_lens_word)}, avg: {avg(q_lens_word):.1f}, median: {median(q_lens_word):.1f}")
    print(f"  Char length  -- min: {min(q_lens_char)}, max: {max(q_lens_char)}, avg: {avg(q_lens_char):.1f}, median: {median(q_lens_char):.1f}")
    direction_counter = Counter(detect_question_direction(item["question"]) for item in data)
    print(f"\n  Question Direction Distribution:")
    for d, cnt in direction_counter.most_common():
        print(f"    {d:<15} {cnt:>6,}  ({pct(cnt, total_qa)})")
    starts = Counter()
    for item in data:
        first_word = item["question"].strip().split()[0] if item["question"].strip() else ""
        starts[first_word] += 1
    print(f"\n  Top-10 question-starting words:")
    for w, cnt in starts.most_common(10):
        print(f"    '{w}' x {cnt:,}")

    print(f"\n{'[4] ANSWER ANALYSIS':^60}")
    print(SEP2)
    a_lens_word = [word_count(item["answer"]) for item in data]
    a_lens_char = [char_count(item["answer"]) for item in data]
    print(f"  Word length  -- min: {min(a_lens_word)}, max: {max(a_lens_word)}, avg: {avg(a_lens_word):.1f}, median: {median(a_lens_word):.1f}")
    print(f"  Char length  -- min: {min(a_lens_char)}, max: {max(a_lens_char)}, avg: {avg(a_lens_char):.1f}, median: {median(a_lens_char):.1f}")
    ans_cat = Counter(detect_answer_category(item["answer"]) for item in data)
    print(f"\n  Answer Trend Category Distribution:")
    for cat, cnt in ans_cat.most_common():
        print(f"    {cat:<15} {cnt:>6,}  ({pct(cnt, total_qa)})")
    unique_answers = len(set(item["answer"].lower().strip() for item in data))
    print(f"\n  Unique answer strings    : {unique_answers:,}  ({pct(unique_answers, total_qa)} of total)")

    print(f"\n{'[5] DATA QUALITY CHECKS':^60}")
    print(SEP2)
    empty_q = sum(1 for item in data if not item.get("question", "").strip())
    empty_a = sum(1 for item in data if not item.get("answer", "").strip())
    print(f"  Empty questions          : {empty_q}")
    print(f"  Empty answers            : {empty_a}")
    numeric_ans = sum(1 for item in data if re.fullmatch(r"[\d\s.,]+", item.get("answer", "").strip()))
    print(f"  Numeric-only answers     : {numeric_ans}  ({'needs review' if numeric_ans > 0 else 'OK'})")
    qa_pairs = [(item["question"].strip().lower(), item["answer"].strip().lower()) for item in data]
    dup_pairs = total_qa - len(set(qa_pairs))
    print(f"  Duplicate QA pairs       : {dup_pairs}  ({'warning' if dup_pairs > 0 else 'OK'})")
    short_ans = sum(1 for item in data if word_count(item.get("answer", "")) < 3)
    print(f"  Answers < 3 words        : {short_ans}  ({'needs review' if short_ans > 5 else 'OK'})")
    long_ans = sum(1 for item in data if word_count(item.get("answer", "")) > 15)
    print(f"  Answers > 15 words       : {long_ans}  ({'exceeds limit' if long_ans > 0 else 'OK'})")
    if os.path.exists(img_dir):
        img_files  = set(f for f in os.listdir(img_dir) if f.endswith(".jpg"))
        missing_imgs = set(item["image"] for item in data) - img_files
        print(f"  Missing image files      : {len(missing_imgs)}  ({'warning' if missing_imgs else 'OK'})")
        if missing_imgs and len(missing_imgs) <= 10:
            print(f"    {list(missing_imgs)}")
    else:
        print(f"  Image dir not found: {img_dir}")
    if os.path.exists(md_dir):
        md_files    = set(f.replace(".md", "") for f in os.listdir(md_dir) if f.endswith(".md"))
        missing_mds = set(item["id"] for item in data) - md_files
        print(f"  Missing markdown files   : {len(missing_mds)}  ({'warning' if missing_mds else 'OK'})")
    else:
        print(f"  Markdown dir not found: {md_dir}")

    print(f"\n{'[6] PER-TYPE QUESTION DIRECTION BREAKDOWN':^60}")
    print(SEP2)
    type_direction = defaultdict(Counter)
    for item in data:
        type_direction[item.get("type", "unknown")][detect_question_direction(item["question"])] += 1
    directions = ["Trend", "Correlation", "Extremum", "Comparison", "Volatility", "Other"]
    print(f"  {'Type':<8}" + "".join(f"{d:>13}" for d in directions))
    print(f"  {'-'*80}")
    for t in ["bar", "line", "area", "pie"]:
        print(f"  {t:<8}" + "".join(f"{type_direction[t].get(d, 0):>13,}" for d in directions))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ViChartVQA Dataset Analyzer")
    parser.add_argument("--dataset_path", default=DEFAULT_DATASET)
    parser.add_argument("--img_dir",      default=DEFAULT_IMG_DIR)
    parser.add_argument("--md_dir",       default=DEFAULT_MD_DIR)
    args = parser.parse_args()

    analyze(
        dataset_path=args.dataset_path,
        img_dir=args.img_dir,
        md_dir=args.md_dir,
    )