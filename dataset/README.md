# ViChartVQA — Vietnamese Chart Visual Question Answering Dataset

An **AI-generated VQA dataset for Vietnamese charts**. Each sample consists of a chart image, a Vietnamese analytical question, and a trend-descriptive answer — no raw number extraction required.

---

## Overview

| Attribute | Details |
|-----------|---------|
| Language | Vietnamese |
| Chart Types | Bar, Line, Area, Pie |
| QA pairs per chart | 4 |
| Image format | JPG (150 DPI, 10×6 inch) |
| Data source | AI-generated (OpenRouter LLM) |
| Export format | JSON + Parquet (HuggingFace-compatible) |
| HuggingFace Hub | [Zenng2812/vqa-vietnamese-charts](https://huggingface.co/datasets/Zenng2812/vqa-vietnamese-charts) |

---

## HuggingFace Dataset

The dataset is publicly available on HuggingFace Hub at:
**[Zenng2812/vqa-vietnamese-charts](https://huggingface.co/datasets/Zenng2812/vqa-vietnamese-charts)**

Load it directly with the `datasets` library:

```python
from datasets import load_dataset

dataset = load_dataset("Zenng2812/vqa-vietnamese-charts")

print(dataset)
# DatasetDict({
#     train:      Dataset(...),
#     validation: Dataset(...),
#     test:       Dataset(...)
# })

sample = dataset['train'][0]
print(sample['question'])    # Vietnamese analytical question
print(sample['answer'])      # Trend-descriptive answer
print(sample['chart_type'])  # bar / line / area / pie
sample['image'].show()       # Chart image (PIL.Image)
```

Each split contains the following fields:

| Field | Type | Description |
|-------|------|-------------|
| `image` | PIL.Image | Chart image |
| `chart_id` | string | Chart identifier (e.g. `chart_00042`) |
| `chart_type` | string | Chart category (`bar`, `line`, `area`, `pie`) |
| `question` | string | Vietnamese analytical question |
| `answer` | string | Trend-descriptive answer |

**Split ratio: 80% train / 10% validation / 10% test** — split by chart (not by QA pair) to prevent data leakage.

---

## Repository Structure

```
project/
├── main.py                  # Main script for dataset generation
├── llm_service.py           # LLM communication via OpenRouter API
├── chart_generator.py       # Chart rendering and Markdown table generation
├── config.py                # Paths, API keys, and prompt configuration
├── normalize.py             # Text normalization (English months → Vietnamese, punctuation)
├── analyze_dataset.py       # Dataset statistics and quality analysis
├── export_to_parquet.py     # Train/val/test split and Parquet export
├── api.txt                  # API key list (one key per line)
└── vqa_flat_dataset/
    ├── images/              # Chart images (.jpg)
    ├── markdown/            # Tabular data in Markdown format (.md)
    └── vqa_dataset.json     # Main dataset file
```

---

## Dataset Generation Pipeline

The entire pipeline is automated in 4 steps:

```
[1] LLM generates chart config
        ↓
[2] chart_generator renders image + creates Markdown table
        ↓
[3] LLM generates 4 questions + 4 answers from the Markdown table
        ↓
[4] Append to vqa_dataset.json
```

### Step 1 — Generate Chart Config

The LLM receives a chart type (`bar`, `line`, `area`, `pie`) and produces a JSON config containing: a theme, a title, X-axis labels, and data columns with `min/max` value ranges. The data is required to have a clear trend and a realistic narrative — not random noise.

### Step 2 — Render Chart and Generate Markdown Table

`chart_generator.py` uses the config to:
- Sample random values within `[min, max]` for each data column.
- Render the chart using `matplotlib` (thread-safe `Agg` backend) and save as `.jpg`.
- Export the data as a Markdown table, used as context for the next step.

### Step 3 — Generate Questions and Answers

The LLM reads the Markdown table and generates **4 analytical questions** across 5 predefined directions:

| Direction | Description |
|-----------|-------------|
| **Trend** | Overall direction (increasing / decreasing / stable) |
| **Correlation** | Relationship between two variables |
| **Extremum** | Identifying peaks, troughs, or notable periods |
| **Comparison** | Comparing two groups or time periods |
| **Volatility** | Degree of fluctuation in the data |

Answers must **describe a trend** in 5–12 words, without citing specific figures.

✅ Valid: `"Tăng dần đều qua các quý"` *(Steadily increasing across quarters)*
❌ Invalid: `"Đạt 85 triệu vào tháng 3"` *(Reached 85 million in March)*

### Step 4 — Storage

Each valid chart (with all 4 QA pairs) is appended to `vqa_dataset.json`. The file is written after every chart to prevent data loss on interruption.

---

## Installation

```bash
pip install pandas matplotlib requests tqdm datasets pillow tabulate
```

---

## Configuration

Open `config.py` and fill in your credentials:

```python
API_KEY    = "your_openrouter_api_key"
MODEL_NAME = "openai/gpt-oss-20b:free"   # or any other model on OpenRouter
```

To use multiple API keys (e.g. to avoid rate limits), add each key on a new line in `api.txt`:

```
sk-or-v1-xxxxxxxxxxxxxxxx
sk-or-v1-yyyyyyyyyyyyyyyy
```

---

## Usage

### Generate Dataset

```bash
python main.py
```

The CLI will prompt for a generation mode:

```
1. Continue (Random)     # Resume from the last chart ID, generate randomly
2. Generate new          # Wipe existing data and generate from scratch
3. Audit & Repair        # Find and fix charts with missing QA pairs
4. Balance dataset       # Generate more samples for a specific chart type
```

### Normalize Text

After generation, run normalization to clean questions and answers (convert English month names to Vietnamese, fix punctuation, etc.):

```bash
python normalize.py
```

### Analyze Dataset Statistics

```bash
python analyze_dataset.py
python analyze_dataset.py --dataset_path ./vqa_flat_dataset/vqa_dataset.json
```

Output includes: total QA count, chart type distribution, question/answer length statistics, and quality checks (duplicates, empty fields, too-short/too-long answers).

### Export to Parquet (HuggingFace)

```bash
python export_to_parquet.py
```

Splits 80/10/10 by chart (not by QA pair), shuffles with fixed seed `42`, and outputs:

```
vqa_vietnamese_parquet/
├── train.parquet
├── validation.parquet
└── test.parquet
```

Images are embedded directly into the Parquet files — upload these 3 files to a HuggingFace Dataset repository and the dataset is ready to use immediately.

---

## Sample Data Format

```json
{
  "id": "chart_00042",
  "image": "chart_00042.jpg",
  "type": "line",
  "question": "Từ Quý 1 đến Quý 4, doanh thu thay đổi như thế nào?",
  "answer": "Tăng dần đều, đạt đỉnh vào cuối năm."
}
```

---

## Notes

- The dataset is **entirely AI-generated**. All chart values are **simulated** (randomly sampled within `[min, max]` ranges proposed by the LLM) and do not reflect real-world data.
- Answer quality depends on the LLM model selected. It is recommended to run `normalize.py` and `analyze_dataset.py` after each generation batch to verify quality.
- If a rate limit is hit, the system automatically waits 30 seconds and retries with the next key in `api.txt`.
- Use **Audit & Repair** mode to find and fill in charts that are missing QA pairs due to API errors.