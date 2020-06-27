import csv
import argparse
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('-csv', type=str, help='path to csv file')
parser.add_argument('-npy', type=str, help='path to map file')
parser.add_argument('-out', type=str, help='path to output file')
args = parser.parse_args()

# Load CSV
lines = csv.reader(open(args.csv))
next(lines)
data = list(lines)

# Load NPY
map_categories = np.load(args.npy)

# Save output csv
out = [['filename', 'category']]
for filename, old_category in data:
    new_category = map_categories[int(old_category)]
    if new_category != -1:
        out.append([filename, new_category])
csv.writer(open(args.out, 'w')).writerows(out)
