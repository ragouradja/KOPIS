from audioop import tomono
import numpy as np
import re
from numpy.core.numeric import allclose
from prediction_class import *
from mask_rcnn import *
from predict_pu import *
from kopis_solutions import *

import urllib.request
import sys
import os
import time
import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import shutil
import copy


def get_bench_name(bench_file):
    return bench_file.split("/")[2]

def get_bench(bench_file, bench_name, pdb = None, download = False, kopis = False):
    print(bench_file, bench_name)
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
                    # a= 0
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
    all_pos = []
    for domain in list_domains:
        all_pos.append(int(domain.delim[0]))
        all_pos.append(int(domain.delim[1]))
    return min(all_pos), max(all_pos)

def continuous_pdb(pdb_file, chain):
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
    ref1 = bench["min"]
    ref2 = bench["size"]
    prot_obj = Domain([ref1,ref2])
    for i in range(len(predictions)):
        cov = predictions[i].get_coverage(prot_obj)
        predictions[i].set_cov_gen(round(cov,2))


def coverage_bench(predictions, bench):
    bench_domain = bench["domain"]
    size = bench["size"]
    min =  bench["min"]
    for i in range(len(predictions)):
        for j in range(len(bench_domain)):
            bench_pred = bench_domain[j].get_coverage(predictions[i])
            if bench_pred >= 85:
                predictions[i].set_bench(bench_domain[j].get_delim())
                predictions[i].set_cov_bench(bench_pred)
        predictions[i].get_info()


def predict_full_bench(all_bench, bench_name, epoch):
    config = PUConfig() # Todo: load model's config from folder
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
    domain_ref = bench_domain["domain"]
    size = bench_domain["size"]
    length_ref = len(domain_ref)
    # if  length_ref == 1:
    #     all_correct += length_ref
    #     return [1], all_correct
    all_correct_value = []
    regex_sol = re.compile("\d+-\d+")
    for sol in solution_list: # sol in full solution list [[]]
        correct = 0
        all_sol_match = regex_sol.findall(" ".join(sol))
        # print(sol, all_sol_match)
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
            # return [0], all_correct
    
    return all_correct_value, all_correct

def bench_bench_get_correct(solution_list, bench_domain, bench_domain_size, all_correct, bench_dict):
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

def final_output(all_bench, bench_name, epoch , list_prot = False):
    nb_domain = 0
    nb_all_domain = 0
    nb_prot = 0
    all_correct = 0
    results_per_domain = {}
    bench_folder = f"../benchmarks/{bench_name}/"
    # bench_file = f"{bench_folder}{bench_name}"
    # all_bench = get_bench(bench_file) # dict
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
            regex_delim = re.compile("[(.*)]")
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
    bench_folder = "../benchmarks/bench/predictions/"

    if not os.path.exists(bench_folder):
        os.mkdir(bench_folder)

    dtf = pd.DataFrame.from_dict(results_per_domain).T
    all_ref = dtf["ref"].sum() 
    all_pred = dtf["pred"].sum()
        


    dtf_correct = dtf["ref"] == dtf["pred"]

    # print(dtf)
    # print(dtf.sum(),dtf.shape)
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
    # f = open("perf.txt","a")

    print(str(epoch) + " "+bench_name + "\n" )
    print(f"Nb correct domain : {all_pred} / {all_ref}: {round(all_pred / all_ref * 100,2)}\n")
    print(f"Nb correct domain assignation: {nb_pred} / {nb_ref} : >{round(nb_pred / nb_ref * 100,2)}% \n\n")
    # f.close()
    # final.to_csv(bench_folder + bench_name + ".csv")      
    # sns.barplot(data=dtf.T.head(10).T)
    # plt.show()
    # plt.savefig("islam.png")
    to_return = [round(nb_pred / nb_ref * 100,2)]
    if list_prot:
        to_return.append(dtf.index.tolist())
    return to_return


def predict_all_benchmarks(epoch):
    folder_bench = "../benchmarks/bench/"
    b = os.listdir(folder_bench)
    done = ["Dissensus"]
    list_prot = {}
    for bench_name in b:
        # if bench_name == "predictions" or "Diss" in bench_name:
        #     continue
        if bench_name not in ["Jones_original_clean","Islam90_original_clean"]:
            continue
        print(f"Kopis vs {bench_name}")
        # if "Dissensus_Cath" == bench_name:
        start = time.time()
        bench_file = f"{folder_bench}{bench_name}"
        
        # bench_name = get_bench_name(bench_file)
        folder_pdb = f"../benchmarks/{bench_name}/"

        if not os.path.exists(folder_pdb):
            os.mkdir(folder_pdb)

        all_bench = get_bench(bench_file, bench_name, download= False, kopis = True)
        # if bench_name not in done:
        #     predict_full_bench(all_bench, bench_name, epoch)
        prot_done, pourc = final_output(all_bench, bench_name, epoch ,list_prot = True)
        list_prot[bench_name] = [pourc, prot_done]
        
        # print(time.time() - start)
    return list_prot

def print_all_csv():
    csv_files = "../benchmarks/bench/predictions"
    files = os.listdir(csv_files)
    for csv in files:
        print(csv)
        print(pd.read_csv(f"{csv_files}/{csv}", index_col=0))
        print()

def compare_benchmarks(jones_islam_dict):
    jones_name = "Jones_original_clean"

    islam_name = "Islam90_original_clean"
    Dissensus_name = "Dissensus_Cath"

    path_folder_bench = "../benchmarks/bench/"
    folder_bench  = os.listdir(path_folder_bench)

    all_bench_islam = get_bench(f"{path_folder_bench}{islam_name}",islam_name, download=False)
    all_bench_jones = get_bench(f"{path_folder_bench}{jones_name}",jones_name,download=False)
    # all_bench_diss = get_bench(f"{path_folder_bench}{Dissensus_name}",Dissensus_name, download=False)


    all_bench_islam_crop = {prot: all_bench_islam[prot] for prot in jones_islam_dict[islam_name][0]}
    all_bench_jones_crop = {prot: all_bench_jones[prot] for prot in jones_islam_dict[jones_name][0]}
    # all_bench_diss_crop = {prot: all_bench_jones[prot] for prot in jones_islam_dict[all_bench_diss][0]}

    # all_bench_islam_crop = all_bench_islam
    # all_bench_jones_crop = all_bench_jones


    for bench_file in folder_bench:

        all = 0
        results_per_domain = {}

        count = 0
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
            correct_value, all_correct = bench_bench_get_correct(all_bench_tool[prot]["domain"], all_bench[prot]["domain"],all_bench[prot]["size"] ,all_correct, bench_dict_ref_islam)
            results_per_domain[prot]["pred"] = correct_value
        print(f"{bench_file} vs {bench_name}")
        jones_islam_dict[bench_file] = predict_to_csv(results_per_domain,bench_file)

    plot_results(jones_islam_dict)


def plot_results(results_dict):
    jones_results = []
    islam_results = []
    jones_label = []
    islam_label = []
    for bench_name in results_dict:
        if len(results_dict[bench_name]) == 2:
            if "Jones" in bench_name:
                jones_results.append(results_dict[bench_name][1])
                jones_label.append(bench_name)
            else:
                islam_results.append(results_dict[bench_name][1])
                islam_label.append(bench_name)
        else:
            if "Jones" in bench_name:
                jones_results.append(results_dict[bench_name][0])
                jones_label.append(bench_name)
            else:
                islam_results.append(results_dict[bench_name][0])
                islam_label.append(bench_name)

    plot_hist(jones_label,jones_results, "../figures/Jones_ok_sol_con.png", "Jones_original_clean")
    plot_hist(islam_label,islam_results, "../figures/Islam_ok_sol_con.png", "Islam90_original_clean")

def plot_hist(x,y, output, bench_name):
    length_y = len(y)
    palette = ["green"]  + ["gray"] * (length_y - 1)
    plt.figure(figsize=(15,15))
    plt.bar(range(length_y), y, color = palette)
    plt.ylim(0,100)
    plt.xticks(range(length_y), x, fontsize = 20, rotation = 20)
    plt.yticks(fontsize = 20)
    # plt.xlabel("Benchmarks")
    plt.ylabel("Correct proposition (%)", fontsize = 15)
    plt.title(f"Comparaison KOPIS predictions and other methods to {bench_name} benchmark", fontsize = 20)
    plt.savefig(output)

if __name__ == "__main__":
    # pred = {"rois": [[ 92,  93, 179, 184],
    # [  1,   2,  91,  93],
    # [  1,   1, 177, 178],
        # [113, 114, 150, 149],
        # [148, 147, 178, 179],
    # [ 35,  34,  72,  71],
    # [  2,   1,  36,  35],
    # [ 76,  75, 120, 121],
    # [ 71,  70,  98,  98]], "scores" : [0.99,0.97,0.96,0.94,0.56,0.53,0.5,0.5,0.2]}


    """
    Verifier que bonne taille dans mrcnn 256
    load_image return pad et pas image !
    """
    # for i in range(10,12):
    jones_islam = predict_all_benchmarks(50)
    compare_benchmarks(jones_islam) 


    # path_folder_bench = "../benchmarks/bench/"


    # bench_name = "Jones_DP"
    # all_bench_islam_dp = get_bench(f"{path_folder_bench}{bench_name}",bench_name, download=False)

    # prot_special =   ["1LTSC",
    #             "1PDCA",
    #             "1PPBL",
    #             "1TABI",
    #             "1TFIA",
    #             "1TGSI",
    #             "2BDSA",
    #             "2BPA3",
    #             "2CBHA",
    #             "2MEV4",
    #             "4CPAI",
    #             "4SGBI",
    #             "2IFOA",
    #             "1PPTA",
    #             "1SISA",
    #             "1PNHA",
    #             "2LTNB"]


    # islam_name = "Jones_original_clean"
    # bench = ["Jones_DDAuth","Jones_DDCath","Jones_DDScop","Jones_DP","Jones_PDP","Jones_original_clean","SWORD_Jones"]


    # islam_name = "Islam90_original_clean"
    # bench = ["Islam90_DDAuth","Islam90_DDCath","Islam90_DDScop","Islam90_DP","Islam90_PDP","Islam90_original_clean","SWORD_Islam90"]


    # all_bench_islam = get_bench(f"{path_folder_bench}{islam_name}",islam_name, download=False)


    # for b in ["Islam90_DDScop"]  :
    #     bad = 0
    #     dom = 0
    #     ok_dom = 0
    #     p = 0
    #     ok = 0
    #     print(p)
    #     all_bench_islam_dp = get_bench(f"{path_folder_bench}{b}",b, download=False)

    #     for prot in all_bench_islam:
    #         try:
    #             all_bench_islam_dp[prot]
    #         except:
    #             continue
    #         # if p >= 6:
    #         #     break
    #         count = 0
    #         print(prot)
    #         print("Ref : ",len(all_bench_islam[prot]["domain"]) , all_bench_islam[prot]["domain"], "vs", "Method : ",len(all_bench_islam_dp[prot]["domain"]), all_bench_islam_dp[prot]["domain"])

    #         if len(all_bench_islam[prot]["domain"]) != len(all_bench_islam_dp[prot]["domain"]):
    #             p += 1
    #             print("Incorrect assignation")
    #             print()
    #             continue
    #         dict_ref = bench_to_dict(all_bench_islam[prot]["domain"], {})

    #         skip = False
    #         dom += len(all_bench_islam[prot]["domain"])
    #         if len(all_bench_islam[prot]["domain"]) == len(all_bench_islam_dp[prot]["domain"]) == 1:
    #             count += 1
    #             ok += 1
    #             p += 1
    #             print("Correct assignation")
    #             print()
    #             continue
                
    #         for pred in all_bench_islam_dp[prot]["domain"]:
    #             for ref in all_bench_islam[prot]["domain"]:
    #                 if dict_ref[ref]:
    #                     uncov = ref.get_coverage(pred)
    #                     print("Ref : {:15} Method : {:15} Cov  (1 - ({} / {})) * 100 = {:5.2f}".format(ref.name,pred.name,ref.get_coverage(pred),all_bench_islam[prot]["size"],float(uncov)))            

    #                     if uncov >= 84:                    
    #                         count += 1
    #                         dict_ref[ref] = 0
    #                         break            
    #         if count == len(all_bench_islam[prot]["domain"]):
    #             print("Correct assignation")
    #             print(len(all_bench_islam[prot]["domain"]) ,len(all_bench_islam_dp[prot]["domain"]))
    #             ok+= 1
    #         else:
    #             print("Incorrect assignation")
    #             print(len(all_bench_islam[prot]["domain"]) ,count)

    #         print()
    #         ok_dom += count
    #         p += 1


    #         print()

    #     print(ok,"/",p, "=", ok /  p * 100 , "%", b)
    #     # print_all_csv()
    
    """
    Bench :  Jones_original_clean
    Nb correct domain : 19 / 30
    Nb domain tested : 30 / 106
    Nb prot tested : 23 / 55

            Reference  Predicted      %
    Domains
    1               19         14  73.68
    2                4          3  75.00
    3                3          1  33.33
    4                4          0   0.00    
    """

