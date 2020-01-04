export PYTHONPATH=$PWD:$PYTHONPATH
pip3 install pyyaml yacs tqdm colorama matplotlib cython tensorboardX
pip3 install -r requirements.txt 
python3 setup.py build_ext --inplace
python3 tools/demo.py \
    --config experiments/siamrpn_r50_l234_dwxcorr/config.yaml \
    --snapshot experiments/siamrpn_r50_l234_dwxcorr/model.pth
