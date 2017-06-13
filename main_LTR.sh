for cv in 0 1
do
    for z_size in 5 10 100
    do
	CUDA_VISIBLE_DEVICES=0 python train_LTR.py  --cv=$cv --z_size=$z_size
    done
done

#CUDA_VISIBLE_DEVICES=0 python train_LTR.py
