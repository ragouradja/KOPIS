"""Script to test KOPIS performance against Benchmarks"""

import re
import urllib.request
import os
import time
import json
import pandas as pd
import matplotlib.pyplot as plt
import copy
import glob
from scipy.stats import wilcoxon

from prediction_class import *
from mask_rcnn import *
from predict_pu import *

def get_bench_name(bench_file):
    """Get bench name from full path of the bench file"""
    return bench_file.split("/")[2]

def get_bench(bench_file, bench_name, pdb = None, download = False, kopis = False):
    """Convert bench delimitation into a dictionnary of object Domain for each PDB code
    Parameters
    ----------
    bench_file : string
        benchmark file name
    bench_name : string
        benchmark name
    pdb : string 
        Possible name of a specific PDB code to extract frm the benchmark file
    download : Boolean value
        If True, it will download the PDB while reading the benchmark file
    kopis : Boolean value
        If True, the discontinuous delimitation from benchmark file are ignored
        Example of delimitation ignored for KOPIS : 50-61;150-253


    Return
    ------
    all_bench : dict
        Dictionnary with PDB code as key and delimitations from benchmark, the first and last position
        of the PDB as value
    """
    bench_obj = []
    all_bench = {}
    delim = re.compile("(\d+)-(\d+)")
    with open(bench_file) as filin:
        for f in filin:
            if f.startswith("#"): 
                continue
            bench_obj = []
            pdb_name = f[:6].strip()
            folder = f'../benchmarks/{bench_name}/{pdb_name}/'
            pdb_file =  f'{folder}{pdb_name}.pdb'
            all_bench[pdb_name] = {}

            if download:
                if not os.path.exists(folder):
                    os.mkdir(folder)
                if not os.path.exists(pdb_file):
                    print(pdb_file ,"downloaded")
                    urllib.request.urlretrieve(f'https://files.rcsb.org/download/{pdb_name[:-1]}.pdb',pdb_file)
                    continuous_pdb(pdb_file, pdb_name[-1])             

                # if not continuous_pdb(pdb_file, pdb_name[-1]):
                #     # shutil.rmtree(folder)
                #     # print(folder,"deleted")
                #     # continue
            # else:
            #     if not os.path.exists(folder):
            #         continue       
            true = f[8:]
            if kopis:
                match_bench = delim.findall(true)
            else:
                match_bench = true.split()
            for domain in match_bench:
                obj = Domain(delim = domain)
                bench_obj.append(obj)
            if pdb:
                if pdb in f:
                    return bench_obj
            all_bench[pdb_name]["domain"] = bench_obj
            first, last = get_size(bench_obj)
            all_bench[pdb_name]["size"] = last
            all_bench[pdb_name]["min"] = first

    return all_bench


def get_size(list_domains):
    """Get first and last position of a PDB
    Parameters
    ----------
    list_domains : list
        List of Domain object representing the delimitations of one PDB
    
    Return
    ------
    min and max : int
        First and last position of a PDB
    """
    all_pos = []
    for domain in list_domains:
        all_pos.append(int(domain.delim[0]))
        all_pos.append(int(domain.delim[1]))
    return min(all_pos), max(all_pos)

def continuous_pdb(pdb_file, chain):
    """Check if the PDB have gaps or jumps in the residue numbering
    Parameters
    ----------
    pdb_file : str
        PDB filename to open
    
    Return
    ------
    ok : Boolean value
        If the PDB is continuous without gap, it returns True, False otherwise
    """
    content = []
    ok = True
    model = False
    with open(pdb_file) as filin:
        prev_pos = -1
        for line in filin:
            if line.startswith("MODEL"):
                if model:
                    break
                model = True
                print(model)
                continue
            if line.startswith("ATOM"):
                items = line.split()
                if items[4] == chain:
                    content.append(line)
                    if "CA" in line:
                        try:
                            cur_pos = int(items[5])
                        except:
                            cur_pos = int(items[5][:-1])

                        if prev_pos != -1:
                            if prev_pos != cur_pos - 1:
                                ok = False
                        prev_pos = cur_pos
    # print(content)
    f = open(pdb_file ,"w")
    f.writelines(content)
    f.close() 
    if ok:
        return True
    return False

def coverage_prot(predictions, bench):
    """Compute coverage of a predicted domain against the full protein

    Set the coverage of the Domain object against the full protein
    and save it in an object attribute with the set_cov_gen() method

    Parameters
    ----------
    predictions : list
        List of predicted delimitation of one PDB in object format (Domain)
    bench : dict
        Dictionnary containing the PDB code as key and the reference delimitation, the first and last position
        as values (from get_bench())
    
    """
    ref1 = bench["min"]
    ref2 = bench["size"]
    prot_obj = Domain([ref1,ref2])
    for i in range(len(predictions)):
        cov = predictions[i].get_coverage(prot_obj)
        predictions[i].set_cov_gen(round(cov,2))


def coverage_bench(predictions, bench):
    """Compute coverage of a predicted domain against each reference delimitation

    Set the coverage of the Domain object against each reference delimitation
    and save it in an object attribute with the set_cov_bench() method

    Parameters
    ----------
    predictions : list
        List of predicted delimitation of one PDB in object format (Domain)
    bench : dict
        Dictionnary containing the PDB code as key and the reference delimitation, the first and last position
        as values (from get_bench())
    """

    bench_domain = bench["domain"]
    for i in range(len(predictions)):
        for j in range(len(bench_domain)):
            bench_pred = bench_domain[j].get_coverage(predictions[i])
            if bench_pred >= 85:
                predictions[i].set_bench(bench_domain[j].get_delim())
                predictions[i].set_cov_bench(bench_pred)
        predictions[i].get_info()


def predict_full_bench(all_bench, bench_name, epoch):
    """Make prediction on all protein from a benchmark file to get all predicted delimitations
    
    Parameters
    ----------
    all_bench : Dict
        Dictionnary containing the PDB code as key and the reference delimitation, the first and last position
        as values (from get_bench())
    bench_name : str
        Benchmark name
    epoch : int
        Epoch to load
    """

    config = PUConfig()
    model_name = f"../results/real_pad_1024_1048/mask_rcnn_real_pad_1024prot_1048_00{epoch}.h5"
    model_pred = MaskRCNN(mode='inference', model_dir='../results/', config = config )
    model_pred.load_weights(model_name, by_name=True)   
       
    SIZE = config.IMAGE_MAX_DIM
    count = 0
    for pdb_code in all_bench:
        config_name = get_name_model(model_name)
        name_pdb = get_pdb_name(pdb_code)

        folder = f"../benchmarks/{bench_name}/{name_pdb}/"
        first_pos, last_pos = get_first_pos(f"{pdb_code}.pdb", folder)
        if not os.path.exists(folder):
            os.mkdir(folder)

        proba_map = get_map(name_pdb, folder)

        if proba_map.shape[0] > SIZE:
            print(f"Protein {pdb_code} too long for this network : {proba_map.shape[0]} > {SIZE}")
            continue

        results = make_predict(proba_map, model_pred = model_pred)
        predictions = pred_to_object(results, first_pos)

        coverage_bench(predictions, all_bench[pdb_code])

        coverage_prot(predictions, all_bench[pdb_code])
        best = solutions_final(predictions)

        write_output(predictions, folder,name_pdb, bench_name, best)


def benchmark_get_correct(solution_list, bench_domain, all_correct, bench_dict):
    """Compare the predicted delimitation against the reference delimitation
    
    Parameters
    ----------
    solution_list : list
        List of predicted solutions given by KOPIS
    bench_domain : dict
        Dictionnary from a benchmark file (from get_bench())
    all_correct : int
        global number of correct domains for KOPIS
    bench_dict : dict
        Dictionnary of covered domain with 1 (covered) and 0 (uncovered)
        Used to process only uncovered domains

    Return
    ------
    correct : int
        Correct predictions for this PDB
    all_correct : int
        Global correct predictions by KOPIS
    """
    domain_ref = bench_domain["domain"]
    size = bench_domain["size"]
    length_ref = len(domain_ref)
    all_correct_value = []
    regex_sol = re.compile("\d+-\d+")
    for sol in solution_list: # sol in full solution list [[]]
        correct = 0
        all_sol_match = regex_sol.findall(" ".join(sol))
        if len(all_sol_match) == length_ref: # only if same nb of domain
            bench_dict = bench_to_dict(domain_ref, bench_dict) #  Flag for each true domain for cov or not
            for pred_delimitation in all_sol_match: # delim in one sol
                obj_pred = Domain(delim = pred_delimitation)
                for ref in domain_ref:
                    if bench_dict[ref]:
                        cov = ref.get_coverage(obj_pred)
                        # if coverage_ref_pred >= 85 and coverage_pred_ref >= 85:
                        if cov >= 85:
                            bench_dict[ref] = 0
                            correct += 1
                            break
                        # else:
                        #     print("Ref : {:15} Method : {:15} Cov  (1 - ({} / {})) * 100 = {:5.2f}".format(ref.name,obj_pred.name,ref.get_uncov(obj_pred),bench_domain_size,float(uncov)))            

        all_correct_value.append(correct)
        if correct == length_ref:
            all_correct += length_ref
            break
        else:
            all_correct += correct
    
    return all_correct_value, all_correct

def bench_bench_get_correct(solution_list, bench_domain, all_correct, bench_dict):
    """Compare the other method delimitation against reference delimitation from Jones or Islam benchmark file
    
    Parameters
    ----------
    solution_list : list
        List of solutions given by other method
    bench_domain : dict
        Dictionnary from a benchmark file (from get_bench())
    all_correct : int
        global number of correct domains
    bench_dict : dict
        Dictionnary of covered domain with 1 (covered) and 0 (uncovered)
        Used to process only uncovered domains
    Return
    ------
    correct : int
        Correct proposals for this PDB
    all_correct : int
        Global correct proposals for a given method
        
    """
    length_ref = len(bench_domain)
    correct = 0
    if len(solution_list) == length_ref == 1:
        all_correct += length_ref
        return 1, all_correct
    # if len(solution_list) == length_ref: # only if same nb of domain
    bench_dict = bench_to_dict(bench_domain, bench_dict) #  Flag for each true domain for cov or not
    for pred_delimitation in solution_list: # delim in one sol
        for ref in bench_domain:
            if bench_dict[ref]:
                cov = ref.get_coverage(pred_delimitation)  
                if cov >= 85:
                    bench_dict[ref] = 0
                    correct += 1
                    break

    if correct == length_ref:
        all_correct += length_ref
        return correct, all_correct
    else:
        all_correct += correct
        return 0, all_correct

def performance_kopis(all_bench, bench_name, epoch , list_prot = False):
    """Manage the prediction and comparaison tasks against benchmark

    Parameters
    ----------
    all_correct : int
        global number of correct domains
    bench_name : str
        Benchmark name
    epoch : int
        Epoch to load
    list_prot : Boolean value
        If True, returns the list of protein used by KOPIS for predictions (all proteins are not considered)
        Argument for predict_to_csv() function
    
    Return
    ------
    Call predict_to_csv() to save the 

    """
    nb_domain = 0
    nb_all_domain = 0
    nb_prot = 0
    all_correct = 0
    results_per_domain = {}
    bench_folder = f"../benchmarks/{bench_name}/"
    all_prot = os.listdir(bench_folder)
    nb_all_prot = len(all_prot)

    for prot in all_bench:  
        all_correct_value = []
        if prot == bench_name:
            continue
        if not continuous_pdb(f"{bench_folder}{prot}/{prot}.pdb",prot[-1]):
            continue
        try:
            predict = f"{bench_folder}{prot}/predict.txt"
            f = open(predict, "r")
        except:
            print("Non predictable :",prot)
            # os.system(f"cat {bench_folder}{prot}/predict.txt")
            continue
        nb_all_domain += len(all_bench[prot])
        bench_dict = bench_to_dict(all_bench[prot]["domain"], {}) #  Flag for each true domain for cov or not
        len_dict  = len(bench_dict)
        results_per_domain[prot] = {"ref": len_dict, "pred" : 0}
        nb_prot += 1
        nb_domain += len_dict
        regex_solution =re.compile("(Best solution :) (.*)")
        content = f.readlines()
        solutions = regex_solution.search("".join(content))
        f.close()
        if solutions: # Get solution list
            solution_match = solutions.group(2)
            a = solution_match.replace("[","[\"")
            a = a.replace("]","\"]")
            solution_list = json.loads(f"[{a[2:-2]}]")
            all_correct_value, all_correct = benchmark_get_correct(solution_list, all_bench[prot],  all_correct, bench_dict)
            if len(all_correct_value) == 1:
                results_per_domain[prot]["pred"] = all_correct_value[0]
            elif len(all_correct_value) == 0:

                results_per_domain[prot]["pred"] = 0
            else:
                results_per_domain[prot]["pred"] = max(all_correct_value)


    
    print(
        f"Nb domain tested : {nb_domain} / {nb_all_domain}\n" 
        f"Nb prot tested : {nb_prot} / {nb_all_prot}\n") 

    return predict_to_csv(results_per_domain,bench_name,  epoch = epoch , list_prot = list_prot)



def predict_to_csv(results_per_domain, bench_name, epoch = None , list_prot = False):
    """Save the pandas dataframe of all predictions / proposals of each PDB to csv file
    
    Parameters
    ----------
    results_per_domain : dict
        Dictionnary containing the reference number of domains
        and the predicted number of correct domains for each PDB
    bench_name : str
        Benchmark name
    Epoch : str
        Epoch to print
    list_prot : Boolean value
        If True, returns the list of Proteins used for the comparaison
    
    Return
    ------
    to_return : list
        May be one or two returns in the list
        - The accuracy on this benchmark
        - The list of proteins used
    """


    bench_folder = "../benchmarks/bench/predictions/"
    if not os.path.exists(bench_folder):
        os.mkdir(bench_folder)
    dtf = pd.DataFrame.from_dict(results_per_domain).T
    all_ref = dtf["ref"].sum() 
    all_pred = dtf["pred"].sum()
    dtf_correct = dtf["ref"] == dtf["pred"]
    dtf_correct[dtf_correct == True] = 1
    dtf_correct[dtf_correct == False] = 0
    dtf_correct.to_csv(bench_folder + bench_name + "_correct.csv")
    max_domain = max(dtf["ref"])
    ref_predictions = []
    index_dtf = []
    for nb in range(1,max_domain+1):
        subdtf = dtf[dtf["ref"] == nb]
        sum_ref = subdtf["ref"].sum()
        # sum_pred = subdtf["pred"].sum()
        subdtf_correct = subdtf["pred"] == subdtf["ref"]
        sum_pred = subdtf_correct.sum()

        index_dtf += [str(nb)]
        ref_predictions.append([subdtf.shape[0], sum_pred, round(sum_pred / subdtf.shape[0] * 100,2)])
    final = pd.DataFrame(ref_predictions, columns = ["Reference","Predicted", "%"], index = index_dtf)
    final.index.name = "Domains"
    # nb_ref = final["Reference"].sum()
    # nb_pred = final["Predicted"].sum()

    nb_ref = dtf_correct.shape[0]  
    nb_pred = dtf_correct.sum()

    print(str(epoch) + " " + bench_name + "\n" )
    print(f"Nb correct domain : {all_pred} / {all_ref}: {round(all_pred / all_ref * 100,2)}\n")
    print(f"Nb correct domain assignation: {nb_pred} / {nb_ref} : >{round(nb_pred / nb_ref * 100,2)}% \n\n")
    final.to_csv(bench_folder + bench_name + "_domains.csv")      
    # sns.barplot(data=dtf.T.head(10).T)
    # plt.show()
    # plt.savefig("islam.png")
    to_return = [round(nb_pred / nb_ref * 100,2)]
    if list_prot:
        to_return.append(dtf.index.tolist())
    return to_return


def predict_all_benchmarks(epoch = 50, predict = False):
    """Main function for predict on benchmark file and compare reference
    domain against the predicted ones


    Parameters
    ----------
    Epoch : int
        Epoch to load (epoch 50 by default)
    predict : Boolean
        If True, make the prediction of all PDB code in the benchmark
    Return
    ------
    list_prot : dict
        Dictionnary of proteins used for Jones and Islam benchmark
    """

    folder_bench = "../benchmarks/bench/"
    b = os.listdir(folder_bench)
    list_prot = {}
    for bench_name in b:
        # if bench_name == "predictions" or "Diss" in bench_name:
        #     continue
        if bench_name not in ["Jones_original_clean","Islam90_original_clean"]:
            continue
        print(f"Kopis vs {bench_name}")
        bench_file = f"{folder_bench}{bench_name}"
        folder_pdb = f"../benchmarks/{bench_name}/"
        if not os.path.exists(folder_pdb):
            os.mkdir(folder_pdb)
        all_bench = get_bench(bench_file, bench_name, download= False, kopis = True)
        if predict:
            predict_full_bench(all_bench, bench_name, epoch)
        prot_done, pourc = performance_kopis(all_bench, bench_name, epoch ,list_prot = True)
        list_prot[bench_name] = [pourc, prot_done]
        
    return list_prot



def compare_benchmarks(jones_islam_dict):
    """Compare the delimitations from other method against Jones or Islam benchmark delimitation
    
    Parameters
    ----------
    jones_islam_dict : dict
        Dictionnary of protein used in both benchmark
    
    """

        
    jones_name = "Jones_original_clean"
    islam_name = "Islam90_original_clean"

    path_folder_bench = "../benchmarks/bench/"
    folder_bench  = os.listdir(path_folder_bench)

    all_bench_islam = get_bench(f"{path_folder_bench}{islam_name}",islam_name, download=False)
    all_bench_jones = get_bench(f"{path_folder_bench}{jones_name}",jones_name,download=False)

    all_bench_islam_crop = {prot: all_bench_islam[prot] for prot in jones_islam_dict[islam_name][0]}
    all_bench_jones_crop = {prot: all_bench_jones[prot] for prot in jones_islam_dict[jones_name][0]}

    for bench_file in folder_bench:
        results_per_domain = {}
        if "original" in bench_file or "Diss" in bench_file or "predictions" in bench_file:
            continue
        all_bench_tool = get_bench(f"{path_folder_bench}{bench_file}",bench_file, download=False)
        if "Islam" in bench_file:
            all_bench = copy.copy(all_bench_islam_crop)
            bench_name = islam_name
        elif "Jones" in bench_file:
            all_bench = copy.copy(all_bench_jones_crop)
            bench_name = jones_name
        for prot in all_bench:
            try:
                all_bench_tool[prot]
            except:
                continue
            bench_dict_ref_islam = bench_to_dict(all_bench[prot]["domain"], {}) #  Flag for each true domain for cov or not
            results_per_domain[prot] = {"ref": len(bench_dict_ref_islam), "pred" : 0}

            all_correct = 0
            correct_value = []
            correct_value, all_correct = bench_bench_get_correct(all_bench_tool[prot]["domain"], all_bench[prot]["domain"] ,all_correct, bench_dict_ref_islam)
            results_per_domain[prot]["pred"] = correct_value
        print(f"{bench_file} vs {bench_name}")
        jones_islam_dict[bench_file] = predict_to_csv(results_per_domain,bench_file)

    data_to_plot(jones_islam_dict)


def data_to_plot(results_dict):
    """Reformating data for plot"""
    results_jones = {}
    results_islam = {}

    for bench_name in results_dict:
        if len(results_dict[bench_name]) == 2:
            if "Jones" in bench_name:
                results_jones[bench_name.split("_")[-1]] = results_dict[bench_name][1]
            else:
                results_islam[bench_name.split("_")[-1]] = results_dict[bench_name][1]
        else:
            if "Jones" in bench_name:
                results_jones[bench_name.split("_")[-1]] = results_dict[bench_name][0]
            else:
                results_islam[bench_name.split("_")[-1]] = results_dict[bench_name][0]

    plot_hist(results_jones, "../figures/Jones_ok_sol_con.png", "Jones_original_clean")
    plot_hist(results_islam, "../figures/Islam_ok_sol_con.png", "Islam90_original_clean")

def plot_hist(results, output, bench_name):
    """Plot KOPIS and other method performances against benchmarks"""
    new_x = ["clean",
            "SWORD",
            "PDP",
            "DP",
            "DDScop",
            "DDAuth",
            "DDCath"]
    new_results = {key:results[key] for key in new_x}
    y = new_results.values()
    x = list(new_results.keys())
    x[0] = "KOPIS"

    print(results)
    print(new_results)
    length_y = len(y)
    palette = ["green"]  + ["blue"] * (length_y - 1)
    plt.figure(figsize=(20,20))
    plt.bar(range(length_y), y, color = palette)
    colors = {'KOPIS':'green', 'Others Methods':'blue'}         
    labels = list(colors.keys())
    handles = [plt.Rectangle((0,0),1,1, color=colors[label]) for label in labels]
    plt.legend(handles, labels, fontsize = 20)
    plt.ylim(0,100)
    plt.xticks(range(length_y), x, fontsize = 30)
    plt.yticks(fontsize = 30)
    # plt.xlabel("Benchmarks")
    plt.ylabel("Correct proposition (%)", fontsize = 25)
    plt.xlabel("Method", fontsize = 25, labelpad=25)
    plt.title(f"{bench_name.split('_')[0].capitalize()} benchmark results for KOPIS predictions", fontsize = 30)
    plt.savefig(output)



def print_all_csv():
    csv_files = "../benchmarks/bench/predictions"
    files = os.listdir(csv_files)
    for csv in files:
        print(csv)
        print(pd.read_csv(f"{csv_files}/{csv}", index_col=0))
        print()

def wilcoxon_test():
    """Make the wilcoxon test with csv file containing the proposals delimitations for each method"""
    csv_folder = "../benchmarks/bench/predictions/"
    jones = glob.glob(f"{csv_folder}Jones*correct.csv")
    islam = glob.glob(f"{csv_folder}Islam*correct.csv")
    jones_original = glob.glob(f"{csv_folder}Jones*ori*correct.csv")
    islam_original = glob.glob(f"{csv_folder}Islam*ori*correct.csv")

    jones_ori_dtf = pd.read_csv(jones_original[0])["0"].values
    islam_ori_dtf = pd.read_csv(islam_original[0])["0"].values

    for path in jones:
        jones_dtf = pd.read_csv(path)["0"].values
        if path == jones_original[0]:
            continue
        print(wilcoxon( jones_ori_dtf,  jones_dtf), path)
    print()
    for path in islam:
        islam_dtf = pd.read_csv(path)["0"].values
        if path == islam_original[0]:
            continue
        print(wilcoxon( islam_ori_dtf,  islam_dtf), path)


def bench_to_dict(bench, bench_domain):
    for domain in bench:
        bench_domain[domain] = 1
    return bench_domain

if __name__ == "__main__":

    # Get prediction and performance for KOPIS against benchmarks
    # Save the protein used by KOPIS and to be use by other method
    jones_islam = predict_all_benchmarks(50, predict = False)
    # Compare other method delimitation against benchmarks with protein used by KOPIS
    compare_benchmarks(jones_islam) 
