from audioop import tomono
from email.base64mime import body_decode
import sys
sys.path.append("../scripts")
from benchmark_test import *
import os


bench_folder = "./bench/"
bench_files = os.listdir(bench_folder)
length = []
poubelle = []
for bench_name in bench_files:    
    if "Diss" in bench_name or "pred" in bench_name or "Jones" in bench_name:
        continue
    print(bench_name)

    bench_file = f"{bench_folder}{bench_name}"
    all_bench, to_move = get_bench(bench_file, bench_name)
    poubelle += to_move
    length.append((len(all_bench),bench_name))
print(length)
print(len(set(poubelle)), len(all_bench) + len(to_move))

bench = open("./bench/Islam90_original")
content = []
for line in bench:
    pdb_name = line[:6].strip()
    if pdb_name in poubelle:
        continue
    content.append(line)

bench.close()

new_bench = open("./bench/Islam90_original_clean","w")
new_bench.writelines(content)
new_bench.close()