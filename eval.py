import os
import sys
import json
from tqdm import tqdm

corr = []
path_dir = sys.argv[1]
for file in tqdm(os.listdir(path_dir)): 
    if file.endswith("json"):
        with open(os.path.join(path_dir, file), "r") as f:
            data = json.load(f)
        corr.append(data[-1]["correct"])
print(round(sum(corr)/len(corr)*100, 3), sum(corr), len(corr))