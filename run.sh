#!/bin/bash
nohup python text2vec.py > splitDatasetAndText2Vector.log 
nohup python train_stageI_instance.py > stageI.log
nohup python train_stageII_rank_instance.py > stageII.log

