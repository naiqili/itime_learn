uif_path='./data/test/uif/uif_test_0.npy'
iur_path='./data/test/iur/iur_test_0.npy'
record_path='./data/test/tfrecord/test_0.record'
bestmodel_dir='./model/test/'
save_path='./log/test/prediction/'

mkdir -p $save_path

python predict.py --config_name='test' --uif_path=$uif_path --iur_path=$iur_path --record_path=$record_path --bestmodel_dir=$bestmodel_dir --save_path=$save_path/prediction.txt --valid_data_size=50
