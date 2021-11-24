import os
import sys
import time
from joblib import Parallel, delayed
from numpy.core.getlimits import _discovered_machar
import numpy as np


def parallel_14000():

    directory = "../../data_peeling/"
    all_prot = next(os.walk(directory))[1]
    os.chdir("../sword/SWORD/")
    start = time.time()
    Parallel(n_jobs= 7, verbose = 0, prefer="processes")(delayed(do_sword)(prot_name) for prot_name in all_prot)
    print("TIME : ",time.time() - start)

def parallel_train():
    os.chdir("../sword/SWORD/")
    f = open("../Benchmarks/pdb_lists/Training.txt")
    all_prot = f.readlines()
    f.close()
    start = time.time()
    Parallel(n_jobs= 7, verbose = 0, prefer="processes")(delayed(do_sword)(prot_name.strip()) for prot_name in all_prot)
    print("TIME : ",time.time() - start)

def parallel(all_prot):
    start = time.time()
    Parallel(n_jobs= 1, verbose = 0, prefer="processes")(delayed(do_sword)(prot_name.strip()) for prot_name in all_prot)
    print("TIME : ",time.time() - start)

def log_to_res(logfile):
        all_coords = []
        with open(logfile, encoding='utf-8') as filin:
            for line in filin:
                if not line.startswith("#"):
                    clean_coord = line.strip().split()[5:]
                    xy = [[int(clean_coord[x]),int(clean_coord[x+1])] for x in range(0,len(clean_coord)-1,2)]
                    all_coords.append(xy)
        return all_coords[-1]

def do_sword(prot_name):
    prot = prot_name[:-1]
    os.system(f"./SWORD -i {prot} -d -c {prot_name[-1]}")

def check():
    os.chdir("../sword/SWORD/")
    directory = "PDBs_Clean"
    all_prot = next(os.walk(directory))[1]
    todo = []
    for prot in all_prot:
        try:
            np.loadtxt(f"{directory}/{prot}/file_proba_contact.mat")
            # log_to_res(f"{directory}/{prot}/Peeling/Peeling.log")
        except:
            os.system(f"rm -rf {directory}/{prot}")
            todo.append(prot)
            print(prot)
            # os.system(f"./SWORD -i {prot[:-1]} -d -c {prot[-1]}")
    print(len(todo))
    parallel(todo)

def get_prot():
    directory = "../../data_peeling/"
    all_prot = next(os.walk(directory))[1]
    with open("all_prot.txt","w") as filout:
        for prot in all_prot:
            filout.write(f"{prot}\n")


if __name__ == "__main__":
    f = open("all_prot.txt")
    print(f.readlines())