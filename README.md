## Install from conda
```bash
conda create -n masifpni_py3613 python=3.6.13
conda activate masifpni_py3613

conda install -y -c bioconda -c conda-forge biopython biopandas=0.2.3
conda install -y -c conda-forge pymesh2 numpy=1.19.5 pandas==1.1.5
conda install -y -c conda-forge tensorflow=1.9.0 absl-py=1.2.0
pip install sklearn networkx configparser tdqm scikit-learn scipy matplotlib gemmi
```

## Test run
#### Note
Before you run masifPNI, you should specify `out_base_dir` in `test_files/test.graphBind.test1.cfg` to your own output directory.

#### 1. Download pdbs
```bash
python masifPNI.py download -f test_files/test.lst --config test_files/test.graphBind.test1.cfg -n 8 --overwrite_pdb

python masifPNI.py download -l "7RAY_A_BC,7F4Y_AB_TPSQ,7F49_A_B,7N8N_ABCD_IJ,7OTJ_AB_CD,7N6I_ABCDEFGHIJ_KL,7OPE_EFGHIKLNOPQRSTUVWXYabcdfghij10_AB2,7N33_ABCDEF_GHIJKL" --config test_files/test.graphBind.test1.cfg -n 8 --overwrite_pdb

python masifPNI.py download -l '1FEU' --config test_files/test.graphBind.test1.cfg -n 8 --overwrite_pdb
```

#### 2. Prepare files
```bash
python masifPNI.py masifPNI-site dataprep -l 1FEU --config test_files/test.graphBind.test1.cfg --overwrite_pdb --no_batch_run --merge_na_chain --na_type RNA -r 9 -max_n 100

python masifPNI.py masifPNI-site dataprep -f test_files/test.lst --config test_files/test.graphBind.test1.cfg --overwrite_pdb --merge_na_chain --na_type RNA -r 9 -max_n 100
```

#### 3. Training
```bash
python masifPNI.py masifPNI-site train --config test_files/test.graphBind.test1.cfg -n 20 --training_list test_files/train.lst --testing_list test_files/test.lst --na_type RNA --merge_na_chain -r 9.0 -max_n 100
```

#### 4. Prediction
```bash
python masifPNI.py masifPNI-site predict -custom_pdb test_files/1FEU.pdb --config test_files/test.graphBind.test2.cfg -n 8 --na_type RNA --model_dir test_files/nn_models_r9_n100/ --merge_na_chain -r 9 -max_n 100 --no_batch_run

python masifPNI.py masifPNI-site predict -l "1FEU" --config test_files/test.graphBind.test1.cfg -n 8 --na_type RNA --model_dir test_files/nn_models_r9_n100/ --merge_na_chain -r 9 -max_n 100 --no_batch_run

python masifPNI.py masifPNI-site predict -f test_files/test.lst --config test_files/test.graphBind.test1.cfg -n 8 --na_type RNA --model_dir test_files/nn_models_r9_n100/ --merge_na_chain -r 9 -max_n 100
```
