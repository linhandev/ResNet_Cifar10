from pathlib import Path
import os
import json

HERE = Path(__file__).parent.absolute()
strip = lambda v: v.strip().strip('"').strip("'").strip()

configs = open(HERE / "configs.csv", "r").readlines()
keys = configs[0].strip().split(",")[1:]
keys = [k for k in keys]
configs = [[strip(v) for v in c.strip().split(",")[1:]] for c in configs[1:]]
configs = [{k: v for k, v in zip(keys, c)} for c in configs if c[0] != ""]

prev_configs = [json.loads(p.read_text()) for p in (HERE / ".." / "output").glob("*/configs.json")]
prev_configs = [{k: str(c.get(k.replace("-", "_"), "")) for k in keys} for c in prev_configs]

for idx, conf in enumerate(configs):
    print("===========================================")
    if conf in prev_configs:
        print(f"Found previous training record for config at line {idx+1}, skipping following")
        print(json.dumps(conf, indent=4))
    else:
        args = [f"--{k} {v}" for k, v in conf.items() if v != ""]
        cmd = f"python {str(HERE/'..'/'train.py')} " + " ".join(args)
        print(f"Training command: {cmd}")
        os.system(cmd)
