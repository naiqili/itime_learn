# Data Preparation

1. Use rival to generate cross validation data /seed2048.
2. Run itimerec/utils.py to process indices.
3. Run RankSys-examples/edu.unimelb.itimerec.examples.BuildLearningData to generate features.
4. Run RankSys-examples/edu.unimelb.itimebot.evaluator.GroundTruthConstructor to generate ground truth ranking.
5. Run itime_learning/data_prepare.py to generate final data.

# Evaluation

1. RankSys-examples/edu.unimelb.itimerec.evaluator.BaselineEvaluator
1. RankSys-examples/edu.unimelb.itimerec.evaluator.SampleEvaluator
