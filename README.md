# BERT-CL
Continual Learning using BERT

pretrain_decoder.py is a try of adding decoder to each layer during continual pre-training

pretrain_stage0.py is the pretrain from zero
pretrain_stage1.py is the pretrain after stage0, with some layers replayed
between stage0 & 1, run get_samples.py to get layer outputs from model after stage0

pretrain_stage1_pure.py is the pretrain after stage0, without any other operation, just pure continual pretrain
