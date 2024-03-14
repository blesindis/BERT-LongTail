from model_bert import Bert
from transformers import BertConfig
import torch
from transformers import AutoTokenizer
from Dataset import WikitextForBert
from accelerate import Accelerator
import torch.nn as nn
import matplotlib.pyplot as plt
# config = BertConfig.from_json_file('/home/archen/privacy/config/bert.json')
# model = Bert(config)
# dataset = WikitextForBert(config)
# device = 'cuda:5'

def get_ntk(model, loader, device = 'cuda:5'):
    
    model = model.to(device)

    grads = {}
    NTKs = {}
    NTKs_norm = {}
    Lambdas = {}
    Lambdas_norm = {}
    

    for i, batch in enumerate(loader):
        if i >= 12: break
        input_ids, attention_mask, labels = batch['input_ids'].to(device), batch['attention_mask'].to(device), batch['labels'].to(device)
        _, logits = model(input_ids, attention_mask, labels)
        # logits = logits.view(-1, config.vocab_size)
        for i in range(len(logits)):
            logits[i:i+1].backward(torch.ones_like(logits[i:i+1]), retain_graph=True)

            for name, param in model.named_parameters():
                if not 'encoder' in name: continue # ignore embedding layers
                if param.requires_grad:
                    grad = grads.get(name, [])
                    grad.append(param.grad.detach().reshape(-1).clone())
                    # print(grad[-1].shape)
                    grads[name] = grad
                
            
            model.zero_grad()
            torch.cuda.empty_cache()

    for name, grad in grads.items():
        J_layer = torch.stack(grad)
        # print(J_layer.shape)
        J_layer_norm = J_layer.T / torch.norm(J_layer.T, dim=0)
        NTKs_norm[name] = J_layer_norm.T @ J_layer_norm
        NTK = J_layer @ J_layer.T
        NTKs[name] = NTK
        u, s, v = torch.linalg.svd(NTKs[name].cpu())
        Lambdas[name] = s
        u, s, v = torch.linalg.svd(NTKs_norm[name].cpu())
        Lambdas_norm[name] = s
        

    return NTKs, NTKs_norm, Lambdas, Lambdas_norm



if __name__ == "__main__":
    config = BertConfig.from_json_file('/home/archen/privacy/config/bert.json')
    model = Bert(config)
    dataset = WikitextForBert(config)
    train_loader = dataset.train_loader
    device = 'cuda:5'
    
    for step in range(0, 60100, 100):
        print(step)
        model_path = f'/home/archen/privacy/checkpoints/bert/step_wiki2_{step}.pth'
        model.load_state_dict(torch.load(model_path))
        NTKs, NTKs_norm, Lambdas, Lambdas_norm = get_ntk(model, train_loader)
        torch.save(NTKs, f'/home/archen/privacy/tmp_result/NTKs_{step}.pt')
        torch.save(NTKs_norm, f'/home/archen/privacy/tmp_result/NTKs_norm_{step}.pt')
        torch.save(Lambdas, f'/home/archen/privacy/tmp_result/Lambdas_{step}.pt')
        torch.save(Lambdas_norm, f'/home/archen/privacy/tmp_result/Lambdas_norm_{step}.pt')
    # datas = dict()
    # datas['to_q'] = dict()
    # datas['to_k'] = dict()
    # datas['to_v'] = dict()
    # datas['ffn1'] = dict()
    # datas['ffn2'] = dict()
    # total_step = 30000
    # for step in range(0, total_step+100, 100):
    #     datas[step] = dict()
    #     datas[step]['to_q'] = dict()
    #     datas[step]['to_k'] = dict()
    #     datas[step]['to_v'] = dict()
    #     datas[step]['ffn1'] = dict()
    #     datas[step]['ffn2'] = dict()
    #     step_data = torch.load(f'/home/archen/privacy/tmp_result/Lambdas_norm_{step}.pt')
    #     for n in step_data.keys():
    #         if 'encoder' not in n:
    #             continue
    #         elif 'to_q' in n:
    #             datas[step]['to_q'][n] = step_data[n][0].item()
    #         elif 'to_k' in n:
    #             datas[step]['to_k'][n] = step_data[n][0].item()
    #         elif 'to_v' in n:
    #             datas[step]['to_v'][n] = step_data[n][0].item()
    #         elif 'dense_1' in n:
    #             datas[step]['ffn1'][n] = step_data[n][0].item()
    #         elif 'dense_2' in n:
    #             datas[step]['ffn2'][n] = step_data[n][0].item()
    # module_name = ['to_q', 'to_k', 'to_v', 'ffn1', 'ffn2']
    # # print(datas[100]['to_v'].keys())
    # module_id=4
    # layer_data = []
    # for layer_id in range(12):
    #     layer_data.append([])
    # for step in range(0, total_step+100, 100):
    #     for name in datas[step][module_name[module_id]].keys():
    #         if 'weight' in name:
    #             splited = name.split('.')
    #             l = int(splited[splited.index('layers') + 1])
    #             layer_data[l].append(datas[step][module_name[module_id]][name])
    # for layer_id in range(12):
    #     plt.figure(layer_id)
    #     plt.plot(range(0, total_step+100, 100), layer_data[layer_id])
    #     plt.xlabel('Training Step')
    #     plt.ylabel('Lambda max')
    #     plt.title(module_name[module_id]+'_'+str(layer_id))
    #     plt.savefig('/home/archen/privacy/result/'+module_name[module_id]+'_'+str(layer_id)+'_norm.png')

                
        
            