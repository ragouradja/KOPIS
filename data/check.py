import os


prot = []
with open("test_set.txt") as filin:
    for line in filin:
        prot.append(line.strip())

print(len(prot))

for p in prot:
    os.system(f"mv  data_sword/{p} test")