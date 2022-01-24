import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns



taille = 200
sword_200 = np.load(f"prot_{taille}.npy")
kopis_200 = np.load(f"prot_kopis_{taille}.npy")



taille = 500
sword_500 = np.load(f"prot_{taille}.npy")
kopis_500 = np.load(f"prot_kopis_{taille}.npy")




# taille = 1000
# sword_1000 = np.load(f"prot_{taille}.npy")
# kopis_1000 = np.load(f"prot_kopis_{taille}.npy")

# sword_1000[sword_1000 > 1000 ]= 430
# # print(sword)

# all_array = [sword_200,kopis_200[1:]]

# sns.boxplot(data=  all_array)
# plt.plot()
# plt.savefig("plot.png")
# kopis_500 = [17.22,0.72,1.47,0.71,0.86,2.01,0.91,2.17,0.72,0.7, 0.5,1.5,2,1,0.3,1.4,1.90,2,1.5,1]
sword_500 = [15.28,12.61,15.42,12.45,11.5,13.71,15.74,9.69,12.44,12.48]
print((kopis_500))   
print((sword_500))   
plt.plot( list(range(len(sword_500))), np.cumsum(sword_500),label = "SWORD")
plt.plot(  list(range(len(sword_500))),np.cumsum(kopis_500),label = "KOPIS")
plt.title("Cumulative sum of execution time SWORD vs KOPIS (500 residues) ", fontsize = 20)
plt.xlabel("Run", fontsize = 20)
plt.ylabel("Time (s)", fontsize = 20)
plt.legend(fontsize = 20)
plt.xticks(fontsize = 20)
plt.yticks(fontsize = 20)
plt.show()
