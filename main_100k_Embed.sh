embed_size=20

for cv in 0 
do
    for z_size in 10
    do
	for keep_prob in 0.9
	do
	    python train_Embed.py  --config_name='100k_embed_main_with_feat' --cv=$cv --z_size=$z_size --keep_prob=$keep_prob --embed_size=$embed_size
	done
    done
done

#CUDA_VISIBLE_DEVICES=0 python train_LTR.py
