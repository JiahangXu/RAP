import os
import sys
import json
from tqdm import tqdm
from collections import defaultdict, Counter

def most_common_element(lst):
    if len(lst) == 0:
        return None
    count = Counter(lst)
    most_common = count.most_common(1)[0]
    return most_common[0]

corr = []
corr_maj = []
corr_avg = []
path_dir = sys.argv[1]
for file in tqdm(os.listdir(path_dir)): 
    if file.endswith("json"):
        try:
            with open(os.path.join(path_dir, file), "r") as f:
                data = json.load(f)
            corr.append(data[-1]["correct"])
            corr_maj.append(float(most_common_element([item["output"] for item in data])) == float(data[0]["answer"].replace(",", "")))
            corr_avg.append(sum([item["correct"] for item in data])/len(data))
        except:
            print(os.path.join(path_dir, file))
print("las", round(sum(corr)/len(corr)*100, 2), sum(corr), len(corr))
print("maj", round(sum(corr_maj)/len(corr)*100, 2), sum(corr_maj), len(corr))
print("avg", round(sum(corr_avg)/len(corr)*100, 2), sum(corr_avg), len(corr))