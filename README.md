# BERT-CL
Continual Learning using BERT


current structure: clustered MoE


warmup.py: output a warm-up tradition bert and cluster centers
pretrain_mix.py: output stage-1 pretrained model
pretrain_mix_new_preparation.py: output new centers(useless now), sample inputs & outputs for each layer
pretrain_mix_new: output stage-2 pretrained model


centers.pth is saved under warm-up model's folder
sample_inputs.pth; sample_outputs.pth are saved under stage-1 pretrained model's folder
