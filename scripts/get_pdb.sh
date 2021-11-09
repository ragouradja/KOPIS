#!/bin/bash

for file in ../data/PDBs_Clean/*;
do 
    directory=../data/`basename $file`
    mkdir -p $directory
    for f in $file/*;
    do
        if  [[ $f =~ ".pdb" ]]; then
            if ! [[ $f =~ "coo" ]]; then
                cp $f $directory
                ../src/distances_MAP_CA $f > $directory/full_prob_map.mat
            fi;
        fi;

        if  [[ $f =~ "Peeling" ]]; then
            cp $f/Peeling.log $directory
        fi;
    done;
done;