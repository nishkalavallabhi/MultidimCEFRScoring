from sklearn.metrics import f1_score
import sys

actual = []
predicted = []

filename = sys.argv[1]
fh = open(filename)

for line in fh:
  temp = line.split("\t")
  actual.append(float(temp[1]))
  predicted.append(float(temp[2]))

print(f1_score(actual,predicted,average='weighted'))
