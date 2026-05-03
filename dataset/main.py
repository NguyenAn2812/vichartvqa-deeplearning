import os
import json
import time
import config
import llm_service
import chart_generator
from tqdm import tqdm


def get_existing_chart_ids():
    if not os.path.exists(config.IMG_DIR):
        return []
    return [f.replace(".jpg", "") for f in os.listdir(config.IMG_DIR) if f.endswith('.jpg')]


def load_dataset():
    if os.path.exists(config.DATASET_FILE):
        try:
            with open(config.DATASET_FILE, "r", encoding="utf-8") as f:
                data = json.load(f)
                return data if isinstance(data, list) else []
        except:
            return []
    return []


def save_dataset(data):
    with open(config.DATASET_FILE, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=4)


def get_type_by_id(chart_id, current_dataset):
    for item in current_dataset:
        if item['id'] == chart_id and 'type' in item:
            return item['type']
    try:
        num = int(chart_id.split('_')[1])
        return "bar" if num <= 279 else "pie"
    except:
        return "bar"


def main():
    dataset      = load_dataset()
    existing_ids = get_existing_chart_ids()

    print(f"Status: {len(existing_ids)} images | {len(dataset)} QA records.")
    print("\n1. Continue (Random)\n2. Generate new\n3. Audit & Repair\n4. Balance dataset")
    choice = input("\nChoice: ")

    tasks       = []
    mode        = 'random'
    forced_type = None

    if choice in ['1', '2', '4']:
        if choice == '2':
            dataset   = []
            start_idx = 1
            num_to_gen = int(input("Number of charts to generate: "))
        elif choice == '4':
            forced_type = input("Type (bar/line/pie/area): ").lower()
            num_to_gen  = int(input(f"Number of {forced_type.upper()} charts: "))
            last_idx    = max([int(cid.split('_')[1]) for cid in existing_ids]) if existing_ids else 0
            start_idx   = last_idx + 1
            mode        = 'fixed'
        else:
            last_idx   = max([int(cid.split('_')[1]) for cid in existing_ids]) if existing_ids else 0
            start_idx  = last_idx + 1
            num_to_gen = int(input(f"Continue from {start_idx}. How many to add: "))

        tasks = [f"chart_{str(i).zfill(5)}" for i in range(start_idx, start_idx + num_to_gen)]

    elif choice == '3':
        counts = {item['id']: 0 for item in dataset}
        for item in dataset:
            counts[item['id']] += 1
        tasks = [cid for cid in existing_ids if counts.get(cid, 0) < 4]
        mode  = 'repair'

    for chart_id in tqdm(tasks, desc="Progress"):
        prefix = f"[{time.strftime('%H:%M:%S')}] [{chart_id}]"
        try:
            if mode == 'repair':
                md_path = os.path.join(config.MD_DIR, f"{chart_id}.md")
                with open(md_path, "r", encoding="utf-8") as f:
                    md_content = f.read()
                chart_type = get_type_by_id(chart_id, dataset)
            else:
                cfg = llm_service.get_single_chart_config(forced_type=forced_type if mode == 'fixed' else None)
                if not cfg:
                    continue
                md_content = chart_generator.create_chart_and_markdown(cfg, chart_id)
                chart_type = cfg.get('type', 'bar')

            questions = llm_service.get_questions(md_content)
            if not questions:
                continue

            answers = llm_service.get_answers_batch(md_content, questions)
            valid_pairs = [
                (q, a) for q, a in zip(questions, answers)
                if q and q.strip() and a and a.strip()
            ]

            if len(valid_pairs) == 4:
                dataset = [item for item in dataset if item['id'] != chart_id]
                for q, a in valid_pairs:
                    dataset.append({
                        "id":       chart_id,
                        "image":    f"{chart_id}.jpg",
                        "type":     chart_type,
                        "question": q.strip(),
                        "answer":   a.strip()
                    })
                save_dataset(dataset)
                print(f"{prefix} Done.")
            else:
                print(f"{prefix} Insufficient QA pairs ({len(valid_pairs)}/4), skipping.")

        except KeyboardInterrupt:
            print("\nStopped.")
            break
        except Exception as e:
            print(f"\nError at {chart_id}: {e}")

    print(f"\nDone. Dataset now has {len(dataset)} records.")


if __name__ == "__main__":
    main()