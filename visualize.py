import argparse
import csv
import os
import shutil

from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument('-d', type=str,
                    help='path to the folder of query images')
parser.add_argument('-c', type=str,
                    help='path to csv file')
parser.add_argument('-o', type=str, default='output',
                    help='output folder (default: output)')
args = parser.parse_args()

reader = csv.DictReader(open(args.c, 'r'))
data = list(reader) 

if os.path.isdir(args.o):
    shutil.rmtree(args.o, ignore_errors=True)
os.mkdir(args.o)
categories = []

for row in tqdm(data):
    filename, category = row['filename'], row['category']

    if category not in categories:
        os.mkdir(os.path.join(args.o, category))
        categories.append(category)

    if (os.path.isfile(os.path.join(args.d, category, filename))):
        input_dir = os.path.join(args.d, category, filename)
    else:
        input_dir = os.path.join(args.d, filename)
    output_dir = os.path.join(args.o, category, filename)

    shutil.copy2(input_dir, output_dir)