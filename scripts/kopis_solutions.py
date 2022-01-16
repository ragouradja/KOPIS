import numpy as np
from prediction_class import *
from benchmark_test import *
import time

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


def bench_to_dict(bench, bench_domain):
    for domain in bench:
        bench_domain[domain] = 1
    return bench_domain

def solutions(predictions, bench):
    domain_ref = bench["domain"]
    size = bench["size"]
    min = bench["min"]
    bench_dict = bench_to_dict(domain_ref, {})
    coverage_prot = 0
    final_solution = []
    temp_solution = []
    N_pred = len(predictions)
    for i in range(N_pred):
        for bench_domain in bench_dict:
            if bench_dict[bench_domain]:
                cov_bench_pred = bench_domain.get_coverage(predictions[i])
                if predictions[i] not in temp_solution:
                    coverage_prot += predictions[i].get_cov_gen()
                    temp_solution.append(predictions[i])
                # print(predictions[i])
                if cov_bench_pred >= 85:
                    bench_dict[bench_domain] = 0
                else:
                    for j in range(i-1,-1,-1):
                        cov_predj_predi = predictions[j].get_coverage(predictions[i])
                        if cov_predj_predi >= 20:
                            continue
                        if predictions[j] not in temp_solution:
                            cov_bench_pred += bench_domain.get_coverage(predictions[j])
                            coverage_prot += predictions[j].get_cov_gen()
                        if cov_bench_pred >= 85:
                            bench_dict[bench_domain] = 0
                            temp_solution.append(predictions[j])
                            break
        if coverage_prot >= 85:
            final_solution.append(sorted(temp_solution))
            temp_solution = []
            coverage_prot = 0
            bench_dict = bench_to_dict(bench["domain"], bench_dict)
    return final_solution


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
                    if cov_pred_domain >= THRESH_COV :
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
    pred = {"rois": [[ 1,  1, 174,174],
    [  85,   85,  169,  169],
    [  1,   1, 87, 87],
    [127, 127,171, 171],
    [81, 81, 133, 133],
    [38, 38, 122, 122],
    [39, 39, 84, 84]], "scores" : [1.0,0.95,0.94, 0.91,0.67,0.62,0.6]}

    bench_file = "../benchmarks/bench/Islam90_original_clean"
    max_length = 171
    pdb_code = "9WGAA"

    bench = get_bench(bench_file, pdb_code)[pdb_code]
    print(bench)
    predictions = pred_to_object(pred,"1")
    coverage_bench(predictions, bench)
    coverage_prot(predictions, bench)
    print_output(predictions, "Jones")
    start = time.time()
    print(solutions(predictions, bench))
    print("time first", time.time() - start)
    start = time.time()

    print(solutions_final(predictions,1,174))
    print("time second", time.time() - start)


    """
    [[1-174], [85-169], [1-87], [127-171], [81-133], [38-122], [39-84]]
    
    2e vrai sol:  [1-87] [85-169]
    
    """