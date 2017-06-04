mkdir -p log/test/
mkdir -p model/test/
mkdir -p log/main/
mkdir -p model/main/

python train_LTR.py  --config_name='test'

#CUDA_VISIBLE_DEVICES=0 python train_LTR.py
