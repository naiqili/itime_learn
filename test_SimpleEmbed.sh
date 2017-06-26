for cv in 0 
do
    for z_size in 10
    do
	for embed_size in 100
	do
	    for keep_prob in 0.5
	    do
		python train_SimpleEmbed.py  --config_name='100k_simple_embed_main_with_feat' --cv=$cv --z_size=$z_size --embed_size=$embed_size --keep_prob=$keep_prob
	    done
	done
    done
done

#CUDA_VISIBLE_DEVICES=0 python train_LTR.py
