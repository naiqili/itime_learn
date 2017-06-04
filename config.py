class Config:
    def __init__(self, config_name="default"):
        if config_name == "default":
            self.get_default_config()
        elif config_name == "test":
            self.get_test_config()

    def get_default_config(self):
        self.user_size = 6040
        self.item_size = 3952
        # self.n_folds = 5
        self.n_folds = 1 # use 1 fold for testing & debugging
        # size of the training/valid data (i.e. size(tfrecord))
        # why the same?
        self.train_data_size = 200042
        self.valid_data_size = 200042

        self.recAlgos = ["pop", "ub", "ib", "hkv", "pzt", "plsa", "lda", "fm-bpr", "fm-rmse"]

        self.recPath = "../RankSys/RankSys-examples/recommendations/learning/"
        self.seed2048Path = "../rival/rival-examples/data/ml-1m/seed2048/"
        self.groundTruthPath = "../RankSys/RankSys-examples/recommendations/"

        self.uifDir = "./data/main/uif/"
        self.iurDir = "./data/main/iur/"
        self.tfrecordDir = "./data/main/tfrecord/"

        # The paths are the uif/iur files to be loaded
        # during training (in LTRModel)
        # When trained on different folds, need to be changed
        # TODO: add valid
        self.train_uif_path = "./data/main/uif/uif_train_0.npy"
        self.train_iur_path = "./data/main/iur/iur_train_0.npy"
        # User test data for validation
        # TODO
        self.valid_uif_path = "./data/main/uif/uif_test_0.npy"
        self.valid_iur_path = "./data/main/iur/iur_test_0.npy"

        self.test_uif_path = ""
        self.test_iur_path = ""

        self.train_record_path = "./data/main/tfrecord/train_0.record"
        self.valid_record_path = "./data/main/tfrecord/test_0.record"
        #self.test_record_path = "./data/main/tfrecord/test_0.record"

        self.lr = 0.001 # learning rate
        self.z_size = 5

        self.max_step = 10000
        self.patience = 100
        self.valid_freq = 50
        self.train_freq = 1

        self.bestmodel_dir = "./model/main/"
        self.log_dir = "./log/main/"
        self.fig_path = "./log/main/"

    def get_test_config(self):
        self.user_size = 500
        self.item_size = 500
        # self.n_folds = 5
        self.n_folds = 1 # use 1 fold for testing & debugging
        # size of the training/valid data (i.e. size(tfrecord))
        # why the same?
        self.train_data_size = 200042
        self.valid_data_size = 20

        self.recAlgos = ["pop", "ub", "ib", "hkv", "pzt", "plsa", "lda", "fm-bpr", "fm-rmse"]

        self.recPath = "../RankSys/RankSys-examples/recommendations/learning/"
        self.seed2048Path = "../rival/rival-examples/data/ml-1m/seed2048/"
        self.groundTruthPath = "../RankSys/RankSys-examples/recommendations/"

        self.uifDir = "./data/test/uif/"
        self.iurDir = "./data/test/iur/"
        self.tfrecordDir = "./data/test/tfrecord/"

        # The paths are the uif/iur files to be loaded
        # during training (in LTRModel)
        # When trained on different folds, need to be changed
        # TODO: add valid
        self.train_uif_path = "./data/test/uif/uif_train_0.npy"
        self.train_iur_path = "./data/test/iur/iur_train_0.npy"
        # User test data for validation
        # TODO
        self.valid_uif_path = "./data/test/uif/uif_test_0.npy"
        self.valid_iur_path = "./data/test/iur/iur_test_0.npy"

        self.test_uif_path = ""
        self.test_iur_path = ""

        self.train_record_path = "./data/test/tfrecord/train_0.record"
        self.valid_record_path = "./data/test/tfrecord/test_0.record"
        #self.test_record_path = "./data/test/tfrecord/test_0.record"

        self.lr = 0.001 # learning rate
        self.z_size = 5

        self.max_step = 10000
        self.patience = 20
        self.valid_freq = 2
        self.train_freq = 1

        self.bestmodel_dir = "./model/test/"
        self.log_dir = "./log/test/"
        self.fig_path = "./log/test/"
