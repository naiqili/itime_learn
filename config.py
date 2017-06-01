class Config:
    def __init__(self):
        self.user_size = 6040
        self.item_size = 3952
        # self.n_folds = 5
        self.n_folds = 1 # use 1 fold for testing & debugging

        self.recAlgos = ["rnd", "pop", "ub", "ib", "hkv", "pzt", "plsa", "lda", "fm-bpr", "fm-rmse"]

        self.recPath = "../RankSys/RankSys-examples/recommendations/learning/"
        self.seed2048Path = "../rival/rival-examples/data/ml-1m/seed2048/"
        self.groundTruthPath = "../RankSys/RankSys-examples/recommendations/"

        self.uifDir = "./data/uif/"
        self.iurDir = "./data/iur/"
        self.tfrecordDir = "./data/tfrecord/"
