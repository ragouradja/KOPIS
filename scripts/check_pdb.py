"""Check if PDB file are without gaps"""

import numpy as np
from numpy.core.numeric import allclose

if __name__ == "__main__":
    data_folder = "../data/"
    pdb_folder = "../data/data_sword/"
    f = open(f"{data_folder}test_set_1024.txt")
    count = 0
    towrite = []
    for prot in f:
        ok = True
        prot = prot.strip()
        path = f"{pdb_folder}{prot}/{prot}.num"
        try:
            res = np.loadtxt(path)
        except:
            continue
        for i in range(1,len(res)):
            if res[i] != res[i-1] + 1:
                count += 1
                ok = False
        if ok:
            towrite.append(prot + "\n")
    f.close()

    f = open(f"{data_folder}test_set_1024_con.txt", "w") 
    f.writelines(towrite)
    f.close()