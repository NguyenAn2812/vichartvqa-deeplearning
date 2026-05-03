import json
import os
import random
import config
from datasets import Dataset, DatasetDict, Image
from PIL import Image as PILImage


def export_to_parquet():
    if not os.path.exists(config.DATASET_FILE):
        print("Dataset JSON not found.")
        return

    with open(config.DATASET_FILE, "r", encoding="utf-8") as f:
        full_data = json.load(f)

    grouped_data = {}
    for item in full_data:
        cid = item['id']
        if cid not in grouped_data:
            grouped_data[cid] = []
        grouped_data[cid].append(item)

    all_chart_ids = list(grouped_data.keys())
    print(f"Total: {len(all_chart_ids)} charts | {len(full_data)} QA records")

    random.seed(42)
    random.shuffle(all_chart_ids)

    total     = len(all_chart_ids)
    train_end = int(total * 0.8)
    val_end   = train_end + int(total * 0.1)

    splits = {
        "train":      all_chart_ids[:train_end],
        "validation": all_chart_ids[train_end:val_end],
        "test":       all_chart_ids[val_end:]
    }

    def build_records(id_list):
        records      = []
        missing_imgs = 0

        for cid in id_list:
            for sample in grouped_data[cid]:
                img_path = os.path.join(config.IMG_DIR, sample['image'])
                if not os.path.exists(img_path):
                    missing_imgs += 1
                    continue
                try:
                    with open(img_path, "rb") as f:
                        img_bytes = f.read()
                    records.append({
                        "image":      {"bytes": img_bytes, "path": sample['image']},
                        "chart_id":   sample['id'],
                        "chart_type": sample.get('type', 'unknown'),
                        "question":   sample['question'],
                        "answer":     sample['answer']
                    })
                except Exception as e:
                    print(f"Image error {img_path}: {e}")

        if missing_imgs:
            print(f"Skipped {missing_imgs} records due to missing images.")

        random.shuffle(records)
        return records

    ds_dict = DatasetDict()
    for split_name, id_list in splits.items():
        print(f"Building {split_name} ({len(id_list)} charts)...")
        records = build_records(id_list)
        ds = Dataset.from_list(records)
        ds = ds.cast_column("image", Image())
        ds_dict[split_name] = ds

    output_dir = "./vqa_vietnamese_parquet"
    os.makedirs(output_dir, exist_ok=True)

    for split_name, ds in ds_dict.items():
        out_path = os.path.join(output_dir, f"{split_name}.parquet")
        ds.to_parquet(out_path)
        size_mb = os.path.getsize(out_path) / 1024 / 1024
        print(f"{split_name}.parquet -- {len(ds):,} records -- {size_mb:.1f} MB")

    print(f"\nDone.")
    print(f"  Train      : {len(splits['train']):>5} charts | {len(ds_dict['train']):>6,} QA")
    print(f"  Validation : {len(splits['validation']):>5} charts | {len(ds_dict['validation']):>6,} QA")
    print(f"  Test       : {len(splits['test']):>5} charts | {len(ds_dict['test']):>6,} QA")


if __name__ == "__main__":
    export_to_parquet()