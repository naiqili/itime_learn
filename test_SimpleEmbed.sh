mkdir -p log/test/
mkdir -p model/test/
mkdir -p log/main/
mkdir -p model/main/

for cv in 0 1
do
    python train_LTR.py  --config_name='test' --cv=$cv
done

#CUDA_VISIBLE_DEVICES=0 python train_LTR.py
