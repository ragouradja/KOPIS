# KOPIS
Kopis is a deep learning tool to predict domains boundaries of a given protein.




Installation
------------
Clone this repository : 

    $ git clone https://github.com/ragouradja/KOPIS.git

Then :

    $ cd KOPIS

Load the conda environment : 

    $ conda env create -f env/kopis_env.yml


Usage
-----

You can run KOPIS with the KOPIS.py script : 

    $ python3 KOPIS.py -p 2ALE -c A -d

This command will download the 2ALE PDB file and run KOPIS on chain A

Remarks
-------

There will be a lot of warning from tensorflow but it dosen't affect KOPIS.
