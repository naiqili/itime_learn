for cv in 0 1
do
    for z_size in 5
    do
	for embed_size in 10
	do
	    python train_SimpleEmbed.py  --config_name='simple_embed_test' --cv=$cv --z_size=$z_size --embed_size=$embed_size
	done
    done
done

#CUDA_VISIBLE_DEVICES=0 python train_LTR.py
