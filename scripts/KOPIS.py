"""Main script to make a prediction with KOPIS

The output will be in
Usage
-----
To download the PDB file : 

python predict_pu.py -p PDB -c [CHAIN] -d

The -c option is not mandatory
Option -p can be either be a file.pdb format or a PDB code to be downloaded with the -p option
"""
import os
import sys
import argparse
import numpy as np
# import pymol
import urllib.request
import shutil

mrcnnpath = "../"
script_path = "."
sys.path += [mrcnnpath, script_path]
from mrcnn.model import MaskRCNN
from mask_rcnn import *
from prediction_class import *
from benchmark_test import *
import warnings
warnings.filterwarnings("ignore")

def get_args():
    """Function to get all args
    Arguments
    ---------
    -p : PDB code or PDB file
    If PDB code is given, need -d option to download the PDB file

    -b : Path to the benchmark contains the reference domains of the PDB
    -c : If a specific chain as to be selected. Otherwise, the whole protein will be selected
    -m : Path to the model to load. Default if MRCNN on 1024 image size at epoch 50
    -d : To download the PDB code without the chain given in -p
    """
    parser = argparse.ArgumentParser(description="Compute prediction of PU and give image and file of PU predicted",
    formatter_class=argparse.RawTextHelpFormatter)

    parser.add_argument("-p", help="Path to the PDB file or PDB code.", type=str, required=True, metavar="pdb")
    parser.add_argument("-b", help="Path to the benchmark file.", type=str, metavar="bench")
    parser.add_argument("-c", help="Chain to select.", type=str, metavar="chain")
    parser.add_argument("-m", help="Path to the model.", type=str, metavar="model",
     default = "../results/real_pad_1024_1048/mask_rcnn_real_pad_1024prot_1048_0050.h5")
    parser.add_argument("-d", help="Boolean to download PDB file.",  action = "store_true")
    args = parser.parse_args()
    
    pdb = args.p
    chain = args.c 
    down = args.d
    model = args.m = rf"{args.m}"
    bench = args.b
    if not os.path.isfile(model):
        print(model)
        sys.exit("Please, enter a valid model.")

    if chain:
        if len(chain) != 1:
            sys.exit("Please enter a valid name of chain")
        args.c = args.c.upper()

    # PDB file on disk
    if not down:
        if "pdb" not in pdb:
            print(pdb)
            sys.exit("Please, enter a valid PDB file.")
        if not os.path.isfile(pdb):
            sys.exit(f"The file {pdb} does not exist. Please enter a valid PDB file.")
        args.p = args.p[:-4].upper() + ".pdb"
    else: # Download PDB file
        try:
            if "pdb" not in pdb:
                args.p = pdb = pdb.upper() + ".pdb"
            urllib.request.urlretrieve(f'https://files.rcsb.org/download/{pdb}', f'{pdb}')
            print(pdb, "downloaded")
        except:
            print(pdb)
            sys.exit("Not a valid PDB code for download")

    if bench:
        if not os.path.isfile(bench):
            sys.exit("Please enter a valid benchmark file.")
    return args



def get_pdb_name(pdb_file):
    """Get pdb name from input path"""
    name_pdb = pdb_file.split(".")
    # Win10
    if name_pdb[0] == "":
        name_pdb = name_pdb[1].split("\\")[1]
    else: # WSL
        name_pdb = name_pdb[0]
    return name_pdb.upper()


def get_chain(pdb_file, chain, folder):
    """Select the chain wanted by the user"""
    content = []
    with open(f"{folder}{pdb_file}") as filin:
        for line in filin:
            if line.startswith("ATOM"):
                if line[21] == chain:
                    content.append(line)
    try:
        first_pos = content[0][22:27].strip()
        last_pos = content[-1][22:27].strip()
    except:
        sys.exit(f"Chain {chain} doesn't exists")

    new_name = pdb_file.split(".")[0] + chain + ".pdb"
    f = open(f"{folder}{new_name}", "w")
    f.writelines(content)
    f.close()
    os.remove(f"{folder}{pdb_file}")
    return new_name, int(first_pos), int(last_pos)

def get_first_pos(pdb_file, folder):
    """Get the first position of the PDB file"""
    first_pos = ""
    last_pos = ""
    content = []
    with open(f"{folder}{pdb_file}") as filin:
        for line in filin:
            if line.startswith("ATOM"):
                content.append(line)
    first_pos = content[0][22:27].strip()
    last_pos = content[-1][22:27].strip()
    try:
        first_pos = int(first_pos)
    except:
        first_pos = int(first_pos[:-1])

    return int(first_pos), int(last_pos)

def get_map(pdb_name, directory):
    """Get the contact probability map by calling a C program 'distance_MAP_CA' """

    if not os.path.exists(directory):  
        os.mkdir(directory)

    output_file = f"{directory}proba_map_{pdb_name}.mat"
    if not os.path.exists(output_file):
        command = f"../src/distances_MAP_CA {directory}{pdb_name}.pdb > {output_file}"
        # command = f"..\src\distances_MAP_CA_win.exe {directory}{pdb_name}.pdb > {output_file}"
        print(command)
        os.system(command)
        print(f"Map written in {output_file}")
    return np.loadtxt(output_file)
    
    
def color_PU(results, first_pos):
    """Color the domains predicted on pymol
    Need to be updated
    """
    # Prendre en compte les gaps !
    list_PU = results["rois"]
    N = len(list_PU)

    for i in range(N):
        PU_name = f"PU_{i+1}"
        x,_,y,_ = list_PU[i]
        x = int(x) +  int(first_pos) - 1
        y = int(y) +  int(first_pos) - 1


        selection = f"resi {x}-{y}"
        pymol.cmd.select(PU_name, selection)
        pymol.cmd.color(color = 'auto', selection = PU_name)

def write_output(predictions, full_folder, name_pdb, bench_name, best = None):
    """Write the output file
    The file contains the domains predicted by KOPIS
    """
    if not os.path.exists(full_folder):
        os.mkdir(full_folder)   
    f = open(f"{full_folder}/predict.txt", "w")
    f.write(f"Predictions for {name_pdb}\n\n")
    if bench_name:
        header = "Prediction\tScore\tCoverage Prot\t\tBenchmark association\t\tBenchmark cov\n"
        f.write(f"Benchmark {bench_name}\n\n")
    else:
        header = "Prediction\tScore\tCoverage Prot\n"

    f.write(header)
    for obj in predictions:
        if bench_name:
            msg = obj.get_info_bench()
        else:
            msg = obj.get_info()
        f.write(msg)
    best = solutions_final(predictions)

    f.write(f'\nBest solution  : \n')
    f.write(f'----------------\n')
    for i,sol in enumerate(best):        
        if i != 0:
            f.write(f"\nAlternative #{i+1} : \n")
            f.write(f'----------------\n')
        for dom in sol:
            f.write(f"{dom.delim[0]}-{dom.delim[1]} ")
        f.write("\n")
    f.close()


def make_predict(image, model_name = None, model_pred = None):
    """Make the predict"""

    config = PUConfig() # Todo: load model's config from folder
    # config.txt_to_config("../results/real_pad_512_1048/real_pad_512_1048.cfg")

    SIZE = config.IMAGE_MAX_DIM
    if image.shape[0] > SIZE:
        sys.exit(f"Protein too long for this network : {image.shape[0]} > {SIZE}")
    image = np.expand_dims(image , axis = 2)
    pad = np.full((SIZE,SIZE,1), dtype = np.float32, fill_value= -1.)
    pad[:image.shape[0],:image.shape[0]] = image
    if model_name:
        model_pred = MaskRCNN(mode='inference', model_dir='../results/', config = config )
        model_pred.load_weights(model_name, by_name=True)   
    r = model_pred.detect([pad])[0]
    return r

def save_fig(name_pdb, full_folder):

    pymol.cmd.png(f"{full_folder}/{name_pdb}_PU.png")
    pymol.cmd.save(f"{full_folder}/{name_pdb}_session.pse")


def get_name_model(model_name):
    if '/' in model_name:
        return model_name.split("/")[2]
    else:
        return model_name.split("\\")[2]


def print_output(predictions, bench_name):
    if bench_name:
        header = "Prediction\tScore\tCoverage Prot\t\tBenchmark association\t\tBenchmark cov\n"
    else:
        header = "Prediction\tScore\tCoverage Prot\n"
    print(header)
    for obj in predictions:
        if bench_name:
            msg = obj.get_info_bench()
        else:
            msg = obj.get_info()
        print(msg)


def solutions_final(pred):
    covered = []
    temp_solution = []
    all_solution = []
    THRESH_COV = 50
    THRESH_GLOBAL = 85
    for i in range(len(pred)):
        temp_solution = []
        cov_global = 0

        take = True
        for domain in temp_solution:
            cov_pred_domain = pred[i].get_coverage(domain)
            if cov_pred_domain >= THRESH_COV:
                take = False
        if take:
            cov_global += pred[i].get_cov_gen()
            temp_solution.append(pred[i])
        if cov_global >= THRESH_GLOBAL:
            all_solution.append(temp_solution)
            temp_solution = []

            cov_global = 0

        else:
            for j in range(i):
                take2 = True
                if pred[j].get_cov_gen() >= 85:
                    continue
                cov_i_j = pred[i].get_coverage(pred[j])
                # print(pred[i],pred[j], cov_i_j, take2)
                if cov_i_j >= THRESH_COV:
                    continue
                for domain in temp_solution:
                    cov_pred_domain = pred[j].get_coverage(domain)
                    if cov_pred_domain >= THRESH_COV:
                        take2 = False
                if take2:
                    cov_global += pred[j].get_cov_gen()
                    temp_solution.append(pred[j])
                    if cov_global >= THRESH_GLOBAL:
                        all_solution.append(sorted(temp_solution))
                        cov_global = 0
                        temp_solution = []
                        break

    return all_solution

if __name__ == "__main__":
    args = get_args()
    pdb_file = args.p # file.pdb
    model_name = args.m
    bench_file = args.b

    name_pdb = get_pdb_name(pdb_file) # file without .pdb
    folder = f"../results/predictions/{name_pdb}/"

    config_name = get_name_model(model_name)
    full_folder = f"{folder}{config_name}/"
    if not os.path.exists(folder):
        os.mkdir(folder)
    
    if not os.path.exists(full_folder):
        os.mkdir(full_folder)
    if args.d:
        shutil.move(pdb_file, f"{full_folder}{pdb_file}")
    else:
        shutil.copy(pdb_file, f"{full_folder}{pdb_file}")

    print(f"mv {pdb_file} {full_folder}{pdb_file} moved")

    if args.c:
        new_pdb_file, first_pos, last_pos = get_chain(pdb_file, args.c, full_folder)
    else:
        new_pdb_file = pdb_file
        first_pos, last_pos = get_first_pos(new_pdb_file, full_folder)
    print(first_pos, last_pos)

    name_pdb = get_pdb_name(new_pdb_file) # file without .pdb



    proba_map = get_map(name_pdb, full_folder)
    results = make_predict(proba_map, model_name)
    predictions = pred_to_object(results, first_pos)

    if bench_file:
        # bench_file = "../benchmarks/Jones_original"
        bench_name = get_bench_name(bench_file)
        bench = get_bench(bench_file, name_pdb)
        coverage_bench(predictions, bench)
    else:
        bench = {"min":first_pos,"size":last_pos}
        bench_name = None

    coverage_prot(predictions, bench)
    write_output(predictions, full_folder,name_pdb, bench_name)
    
    # pymol.cmd.load(new_pdb_file)
    # pymol.cmd.remove("solvent")
    # color_PU(results, first_pos)
    # save_fig(name_pdb, full_folder)
    # pymol.cmd.quit()
    # os.system(f"move {new_pdb_file} {full_folder}")
