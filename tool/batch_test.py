from pathlib import Path
import json
import os
import pandas as pd

HERE = Path(__file__).parent.absolute()


for info_path in (HERE / ".." / "output").glob("*/info.json"):
    info = json.loads(info_path.read_text())
    if "test_loss" in info.keys():
        print(f"{info_path.parent} evaluated, skipping")
        continue
    if list(info_path.parent.glob("*_best.pt")) == []:
        continue
    cmd = f"python {str(HERE / '..' / 'evaluate.py')} --model-name {info['model_name']} --model-save-path {info_path.parent}"
    print(cmd)
    os.system(cmd)


keys = [
    "test_loss",
    "test_accuracy",
    "model_name",
    "num_epoch",
    "batch_size",
    "learning_rate",
    "do_aug",
    "optimizer",
    "scheduler",
    "bs_increase_at",
    "bs_increase_by",
    "loss",
]
infos = []
for info_path in (HERE / ".." / "output").glob("*/info.json"):
    info = json.loads(info_path.read_text())
    if "test_loss" not in info.keys():
        continue
    info["bs_increase_by"] = ",".join(map(str, info["bs_increase_by"]))
    info["bs_increase_at"] = ",".join(map(str, info["bs_increase_at"]))
    infos.append([info.get(k, "-") for k in keys])

print(infos)
infos = pd.DataFrame(infos)
print(infos)
infos.columns = keys

infos.to_csv(HERE / "test_results.csv")
