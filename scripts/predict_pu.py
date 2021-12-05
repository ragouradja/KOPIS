import os
import sys
import argparse
import numpy as np
import pymol
import cv2
import urllib.request

mrcnnpath = "../"
script_path = "."
sys.path += [mrcnnpath, script_path]
from mrcnn.model import MaskRCNN
from mask_rcnn import *

def get_args():

    parser = argparse.ArgumentParser(description="Compute prediction of PU and give image and file of PU predicted",
    formatter_class=argparse.RawTextHelpFormatter)

    parser.add_argument("-p", help="Path to the PDB file.", type=str, required=True, metavar="pdb")
    parser.add_argument("-c", help="Chain to select.", type=str, metavar="chain")
    parser.add_argument("-m", help="Path to the model.", type=str, metavar="model",
     default = "../results/sword12k_pad_heads_13continue_20211125T0105/mask_rcnn_sword12k_pad_heads_13continue__0030.h5")
    parser.add_argument("-d", help="Boolean to download PDB file.",  action = "store_true")

    args = parser.parse_args()
    
    pdb = args.p
    chain = args.c 
    down = args.d
    model = args.m = rf"{args.m}"

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

    else: # Download PDB file
        try:
            if "pdb" not in pdb:
                args.p = pdb = pdb + ".pdb"
            urllib.request.urlretrieve(f'https://files.rcsb.org/download/{pdb}', f'{pdb}')
        except:
            print(pdb)
            sys.exit("Not a valid PDB code for download")

    print(f"File {pdb} downloaded")
    return args



def get_pdb_name(pdb_file):
    name_pdb = pdb_file.split(".")
    # Win10
    if name_pdb[0] == "":
        name_pdb = name_pdb[1].split("\\")[1]
    else: # WSL
        name_pdb = name_pdb[0]
    return name_pdb.upper()


def get_chain(pdb_file, chain):
    content = []
    with open(pdb_file) as filin:
        for line in filin:
            if line.startswith("ATOM"):
                if line[21] == chain:
                    content.append(line)
    try:
        first_pos = content[0][22:27].strip()
    except:
        sys.exit(f"Chain {chain} doesn't exists")

    new_name = pdb_file.split(".")[0] + chain + ".pdb"
    f = open(new_name, "w")
    f.writelines(content)
    f.close()
    os.remove(pdb_file)
    return new_name, first_pos

def get_first_pos(pdb_file):
    with open(pdb_file) as filin:
        for line in filin:
            if line.startswith("ATOM"):
                return line[22:27].strip()
    return first_pos

def get_map(pdb_file, pdb, directory):

    if not os.path.exists(directory):  
        os.mkdir(directory)

    output_file = f"{directory}proba_map_{pdb}.mat"
    if not os.path.exists(output_file):
        # cmd = f"../src/distances_MAP_CA {pdb_file} > {output_file}"
        command = f"..\src\distances_MAP_CA_win.exe {pdb_file} > {output_file}"
        os.system(command)
        print(f"Map written in {output_file}")
    return np.loadtxt(output_file)
    
    
def color_PU(results, first_pos, full_folder):

    # Prendre en compte les gaps !

    if not os.path.exists(full_folder):
        os.mkdir(full_folder)   
    print(full_folder)
    f = open(f"{full_folder}/predict.txt", "w")

    list_PU = results["rois"]
    N = len(list_PU)

    for i in range(N):
        PU_name = f"PU_{i+1}"
        x,_,y,_ = list_PU[i]
        x = int(x) +  int(first_pos) - 1
        y = int(y) +  int(first_pos) - 1
        msg = "PU predicted : {} {}; score : {}".format( x, y, results["scores"][i])
        print(msg)
        f.write(msg + "\n")

        selection = f"resi {x}-{y}"
        pymol.cmd.select(PU_name, selection)
        pymol.cmd.color(color = 'auto', selection = PU_name)
    f.close()


def make_predict(image, model, normalised = 0):

    if normalised:
        image = cv2.cvtColor(np.array(image).astype(np.float32), cv2.COLOR_RGB2BGR)
    else:
        image *= 255
        image = cv2.cvtColor(np.array(image).astype(np.uint8), cv2.COLOR_RGB2BGR)

    config = PUConfig()
    model_pred = MaskRCNN(mode='inference', model_dir='../results/', config = config )
    model_pred.load_weights(model, by_name=True)
    r = model_pred.detect([image])[0]
    return r

def save_fig(name_pdb, full_folder):

    pymol.cmd.png(f"{full_folder}/{name_pdb}_PU.png")
    pymol.cmd.save(f"{full_folder}/{name_pdb}_session.pse")


def get_name_model(model_name):
    if '/' in model_name:
        return model_name.split("/")[2]
    else:
        return model_name.split("\\")[2]
    

if __name__ == "__main__":
    args = get_args()
    pdb_file = args.p
    model_name = args.m

    if args.c:
        pdb_file, first_pos = get_chain(pdb_file, args.c)
    else:
        first_pos = get_first_pos(pdb_file)
        print(first_pos)
    config_name = get_name_model(model_name)
    name_pdb = get_pdb_name(pdb_file)

    folder = f"../results/predictions/{name_pdb}/"
    full_folder = f"{folder}{config_name}"

    proba_map = get_map(pdb_file, name_pdb, folder)
    results = make_predict(proba_map, model_name, normalised=0)

    # pymol.finish_launching()

    pymol.cmd.load(pdb_file)
    pymol.cmd.remove("solvent")
    color_PU(results, first_pos, full_folder)
    save_fig(name_pdb, full_folder)
    pymol.cmd.quit()
    # os.system(f"move {pdb_file} {full_folder}")
