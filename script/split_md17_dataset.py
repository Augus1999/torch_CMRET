import csv
from cmret.utils import split_data

file_name = "paracetamol.xyz"  # the uncompressed file downloaded from http://sgdml.org/

train_idx, test_idx = [], []
with open("split_index/index_train.csv", "r") as g:
    reader = csv.reader(g)
    for i in reader:
        train_idx.append(int(i[0]))
with open("split_index/index_test.csv", "r") as h:
    reader = csv.reader(h)
    for i in reader:
        test_idx.append(int(i[0]))
split_data(file_name=file_name, train_split_idx=train_idx, test_split_idx=test_idx)
