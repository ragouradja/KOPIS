import matplotlib.pyplot as plt


with open("epoch.txt") as filin:
    nb  = 0
    islam = []
    jones = []
    acc = []
    for line in filin:
        if ">" in line:
            acc.append(float(line.split(">")[-1].strip()[:-1]))

print(acc)
islam = acc[::2]
jones = acc[1::2]
x = list(range(10,51))

print(len(x), len(islam))
plt.figure(figsize=(15,10))
plt.plot(x,islam ,label = "islam")
plt.plot(x,jones ,label = "jones")
plt.legend()
plt.title("Accuracy on benchmarks by epoch")
plt.xlabel("Epoch")
plt.ylabel("Accuracy in %")
plt.savefig("acc_refined.png")
plt.show()