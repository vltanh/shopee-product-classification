import argparse
import csv
from pathlib import Path
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('-d', type=str,
                    help='path to the folder of csv files')
parser.add_argument('-o', type=str, default='ensembled_output.csv',
                    help='output file (default: ensembled_output.csv)')
args = parser.parse_args()

sum_probs = np.zeros((12186, 42), dtype=float)
for path in Path(args.d).glob('*.csv'):
    print(f'[PROCESSING] {path}')
    reader = csv.DictReader(open(path, 'r'))
    data = list(reader)

    probs = []
    file_names = []
    for row in data:
        x = [v for k, v in row.items()][2:]
        x = [float(v) for v in x]
        probs.append(x)
        file_names.append(list(row.items())[0][1])

    sum_probs += probs

out = [['filename', 'category']]
for i in range(sum_probs.shape[0]):
    out.append([file_names[i], f'{np.argmax(sum_probs[i]):02d}'])

csv.writer(open(args.o, 'w')).writerows(out)
