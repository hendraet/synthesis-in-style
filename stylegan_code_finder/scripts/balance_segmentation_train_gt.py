import argparse
import json
import random
from collections import defaultdict
from pathlib import Path

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Take train gt for semantic segmentation training and balance it")
    parser.add_argument("gt", help="Path to JSON holding gt to balance")

    args = parser.parse_args()

    with open(args.gt) as f:
        gt_data = json.load(f)

    class_to_file_name_map = defaultdict(list)
    class_keys = [key for key in gt_data[0] if 'has' in key]

    for gt_item in gt_data:
        class_key_matches = [gt_item[key] for key in class_keys]
        if all(class_key_matches):
            class_to_file_name_map['all'].append(gt_item)
        elif not any(class_key_matches):
            class_to_file_name_map['none'].append(gt_item)
        else:
            for key, match in zip(class_keys, class_key_matches):
                if match:
                    class_to_file_name_map[key].append(gt_item)

    items_per_class = {key: len(values) for key, values in class_to_file_name_map.items()}
    smallest_class = min(items_per_class.values())
    print(f"keeping {smallest_class} files per class")

    for key in class_to_file_name_map:
        random.shuffle(class_to_file_name_map[key])

    for key in class_to_file_name_map:
        class_to_file_name_map[key] = class_to_file_name_map[key][:smallest_class]

    kept_items = []
    for items in class_to_file_name_map.values():
        kept_items.extend(items)

    random.shuffle(kept_items)

    args.gt = Path(args.gt)
    dest_file_name = args.gt.parent / f"{args.gt.stem}_balanced.json"
    with dest_file_name.open('w') as f:
        json.dump(kept_items, f)
