# PySOT
This repository is the add-on package for [PySOT](https://github.com/STVIR/pysot) project.

## Install command!!!!!!!!

export PYTHONPATH=$PWD:$PYTHONPATH

pip3 install -r requirements.txt 

python3 setup.py build_ext --inplace

python3 tools/demo.py \
    --config experiments/siamrpn_r50_l234_dwxcorr/config.yaml \
    --snapshot experiments/siamrpn_r50_l234_dwxcorr/model.pth

## Download 
https://drive.google.com/drive/folders/1YbPUQVTYw_slAvk_DchvRY-7B6rnSXP9

mv model.pth experiment/siammask3_r50_l3/model.pth

