import csv

ls = []

with open("names.csv", 'r') as file:
    csvreader = csv.reader(file)
    for row in csvreader:
        ls.append(row)

with open('names.txt', 'w') as f:
    f.write(str(ls))
