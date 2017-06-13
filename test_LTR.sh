mkdir -p log/test/
mkdir -p model/test/
mkdir -p log/main/
mkdir -p model/main/

for cv in 0 1
do
    for z_size in 5
    do
	python train_LTR.py  --config_name='test' --cv=$cv --z_size=$z_size
    done
done

#CUDA_VISIBLE_DEVICES=0 python train_LTR.py
