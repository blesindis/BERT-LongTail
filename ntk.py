import torch

import matplotlib.pyplot as plt
import util
from sklearn.metrics.pairwise import cosine_similarity
import seaborn as sns
from sklearn.decomposition import PCA

import numpy as np
def pca_projection_with_torch(input,projects,k0):
    input = input.cpu().numpy()
    matrix = projects.cpu().numpy()

    pca = PCA(n_components=k0,whiten=True)  # 假设你想要降维到2维
    pca.fit(matrix)

    # 假设你有一个单独的张量
    if not np.allclose(input, np.zeros_like(input)):
        tensor = input / np.linalg.norm(input)
    else:
        tensor = input

    # 使用PCA的转换基来转换张量
    transformed_tensor = torch.tensor(pca.transform(tensor))
    
    norm_t = torch.sum(transformed_tensor)/k0
    # print(norm_t)
    return transformed_tensor, norm_t

def pca_similarity_with_torch(input,projects,k0):
    input = input.cpu().numpy()
    matrix = projects.cpu().numpy()

    pca = PCA(n_components=k0,whiten=True)  # 假设你想要降维到2维
    pca.fit(matrix)
    # explained_variance_ratio = pca.explained_variance_ratio_
    # print(explained_variance_ratio[0],explained_variance_ratio[-1])
    transformed_basis = pca.components_

    # 假设你有一个单独的张量
    if not np.allclose(input, np.zeros_like(input)):
        tensor = input / np.linalg.norm(input)
    else:
        tensor = input
    # 使用PCA的转换基来转换张量
    transformed_tensor = torch.tensor(pca.transform(tensor))

    cosine_similarities = cosine_similarity(tensor, transformed_basis)

    cosine_similarities = torch.tensor(np.mean(cosine_similarities))
    
    return transformed_tensor, cosine_similarities


def get_ntk(model: MoMoE_0126, loader, accelerator: Accelerator,cluster_centers,PRO_VECS,model0,device2):
    
    module = model.module if isinstance(model, nn.parallel.DistributedDataParallel) else model
    config = module.config

    grads = {}
    NTKs_norm = {}
    
    head_grads = [[[] for _ in range(config.num_attention_heads)] for _ in range(config.num_hidden_layers)]
    head_NTKs_norm = [[0 for _ in range(config.num_attention_heads)] for _ in range(config.num_hidden_layers)]
    expert_NTKs_norm = [[0 for _ in range(config.num_experts)] for _ in range(config.num_hidden_layers)]

    for i, batch in enumerate(loader):
        if i >= 1: break
        # batch = {k: v[:1, :] for k, v in batch.items()}
        batch0 = {key: tensor.to(device2) for key, tensor in batch.items()}
        
            
        _,_,_,_,_,_,inputs = model0(**batch0)
        inputs = [i0.to(device) for i0 in inputs]




        _, logits, _, _, _,_,_,_,_,_ = model(batch['input_ids'],batch['attention_mask'], batch['labels'], cluster_centers,inputs,PRO_VECS)
        # logits = logits.view(-1, config.vocab_size)

        for i in range(len(logits)):
            # b, s, v = logits.shape
            # param_grads = torch.autograd.grad(logits[i:i+1], model.parameters(), grad_outputs=torch.ones_like(logits[i:i+1]), create_graph=True)
            logits[i:i+1].backward(torch.ones_like(logits[i:i+1]))
            print(i)
            # accelerator.backward(logits[i:i+1], gradient=torch.ones_like(logits[i:i+1]))

            for name, param in module.named_parameters():
                # print(name)
                if not 'layers' in name: continue # ignore embedding layers
                if param.requires_grad:
                    grad = grads.get(name, [])
                    grad.append(accelerator.gather(param.grad.detach().reshape(-1).unsqueeze(0).clone()))
                    grads[name] = grad
                if 'heads' in name and 'attentions.2' in name:
                    grad = param.grad.detach().flatten()
                    splited = name.split('.')
                    l = int(splited[splited.index('layers') + 1])
                    h = int(splited[splited.index('heads') + 1])
                    # print(name)
                    head_grads[l][h].append(accelerator.gather(grad.unsqueeze(0).clone()))
                    
            
            model.zero_grad()
            torch.cuda.empty_cache()

    for name, grad in grads.items():
        J_layer = torch.concat(grad)

        J_layer_norm = J_layer.T / torch.norm(J_layer.T, dim=0)
        NTK_norm = J_layer_norm.T @ J_layer_norm
        NTKs_norm[name] = NTK_norm

        if 'layers' in name:
            splited = name.split('.')
            l = int(splited[splited.index('layers') + 1])
            if 'experts' in name:
                e = int(splited[splited.index('experts') + 1])
                expert_NTKs_norm[l][e] += NTKs_norm[name]
            
    for l, layer in enumerate(head_grads):
        for h, head in enumerate(layer):
            J_head = torch.concat(head)
            # print(J_head.shape)
            J_head_norm = J_head.T / torch.norm(J_head.T, dim=0)
            # print(J_head_norm.shape)
            # J_head_norm = J_head.T 

            head_NTK_norm = J_head_norm.T @ J_head_norm


            head_NTKs_norm[l][h] += head_NTK_norm.T
        
        head_NTKs_norm[l] = torch.stack(head_NTKs_norm[l])
        # print(head_NTKs_norm[l].shape)
        expert_NTKs_norm[l] = torch.stack(expert_NTKs_norm[l])
    
    head_NTKs_norm = torch.stack(head_NTKs_norm).cpu() # l, h, n, n
    expert_NTKs_norm = torch.stack(expert_NTKs_norm).cpu() # l, e, n, n

    head_lmax_norm = torch.stack([
        torch.stack([util.lmax(head_NTKs_norm[l][h]) 
        for h in range(config.num_attention_heads)])
        for l in range(config.num_hidden_layers)
    ])

    expert_lmax_norm = torch.stack([
        torch.stack([util.lmax(expert_NTKs_norm[l][e]) 
        for e in range(config.num_experts)])
        for l in range(config.num_hidden_layers)
    ])

    return head_lmax_norm, expert_lmax_norm

def get_ntk_by_layer(model,loader,cluster_centers,PRO_VECS,model0,device2,steps,doamin,longtailed_data,o):

    L , W, H = 12,3,8
    NTKs = [[[[] for k in range(H)]for j in range(W)]for i in range(L)]
    longtailed_NTKs = [[[[] for k in range(H)]for j in range(W)]for i in range(L)]
    J_GRADS = [[[[] for k in range(H)]for j in range(W)]for i in range(L)]

    NTK = torch.zeros(L,W,H)
    longtailed_NTK = torch.zeros(L,W,H)
    longtailed_sim = torch.zeros(L,W,H)

    params_that_need_grad = []
    for param in model.parameters():
        if param.requires_grad:
            params_that_need_grad.append(param.requires_grad)
            param.requires_grad = False # first set all gradients to not calculate, time saver
        else:
            params_that_need_grad.append(param.requires_grad)
    
    for index, (name, param) in enumerate(model.named_parameters()):
        if not params_that_need_grad[index]: #if it didnt need a grad, we can skip it.
            continue
        
        if 'heads' in name and 'attentions' in name:
            param.requires_grad = True #we only care about this tensors gradients in the loop
            this_grad = []
            longtailed_data_this_grad = []
            splited = name.split('.')
            l = int(splited[splited.index('layers') + 1])
            t = int(splited[splited.index('attentions') + 1])
            h = int(splited[splited.index('heads') + 1])
            print(l,t,h)
            if l >= L:break
            print(name)

            for i, batch in enumerate(loader):
                if i >= 1: break
                batch0 = {key: tensor.to(device2) for key, tensor in batch.items()}
        
            
                _,_,_,_,_,_,inputs = model0(**batch0)
                inputs = [i0.to(device) for i0 in inputs]
                


                _, logits, _, _, _,_,_,_,_,_ = model(batch['input_ids'],batch['attention_mask'], batch['labels'], cluster_centers,inputs,PRO_VECS)

                for i in range(len(logits)):
                    # print(i)
                    logits[i:i+1].backward(torch.ones_like(logits[i:i+1]), create_graph=True)
                    this_grad.append(param.grad.detach().reshape(-1).clone())
                    model.zero_grad()
                    torch.cuda.empty_cache()

                batch0 = {key: tensor.to(device2) for key, tensor in longtailed_data.items()}
        
                _,_,_,_,_,_,inputs = model0(**batch0)
                inputs = [i0.to(device) for i0 in inputs]
                
                _, logits, _, _, _,_,_,_,_,_ = model(longtailed_data['input_ids'],longtailed_data['attention_mask'], longtailed_data['labels'], cluster_centers,inputs,PRO_VECS)

                for i in range(len(logits)):
                    # print(i)
                    logits[i:i+1].backward(torch.ones_like(logits[i:i+1]), create_graph=True)
                    longtailed_data_this_grad.append(param.grad.detach().reshape(-1).clone())
                    model.zero_grad()
                    torch.cuda.empty_cache()

            J_layer = torch.stack(this_grad) # [N x P matrix] #this will go against our notation, but I'm not adding
            longtailed_J_layer = torch.stack(longtailed_data_this_grad) # [N x P matrix] #this will go against our notation, but I'm not adding
 
            NTKs[l][t][h].append(J_layer @ J_layer.T )# An extra transpose operation to my code for us to feel better
            J_GRADS[l][t][h].append(J_layer)
            # print(NTKs[l][t][h])
            longtailed_NTKs[l][t][h].append(longtailed_J_layer)
            param.requires_grad = False
            model.zero_grad()
            torch.cuda.empty_cache()

            

     
    #reset the model object to be how we started this function
    for i, param in enumerate(model.parameters()):
        if params_that_need_grad[i]:
            param.requires_grad = True
    
    for l in range(L):
        for w in range(W):
            for h in range(H):
                # if len(NTKs[l][w][h]) == 0:
                #     NTK[l][w][h] = torch.tensor(0)
                
                # print(util.lmax(NTKs[l][w][h][0]))
                NTK[l][w][h] = util.lmax(NTKs[l][w][h][0])
                # print(longtailed_NTKs[l][t][h][0])
                _,longtailed_NTK[l][w][h] = pca_projection_with_torch(longtailed_NTKs[l][t][h][0],J_GRADS[l][w][h][0],4)
                # print(longtailed_NTK[l][w][h])
                _,longtailed_sim[l][w][h] = pca_similarity_with_torch(longtailed_NTKs[l][t][h][0],J_GRADS[l][w][h][0],4)
                # print(longtailed_sim[l][w][h])
            # print(NTK[l][w])
            # print(longtailed_NTK[l][w])
            # print(longtailed_sim[l][w])


    lambdamax_layers = NTK

    longtailed_NTK[longtailed_NTK == float('inf')] = 0
    longtailed_sim[longtailed_sim == float('inf')] = 0
    print(torch.max(longtailed_NTK[:,0:2,:], dim=1)[0])

    # print(torch.max(lambdamax_layers[:,0:2,:],dim=1)[0].shape)
    plt.figure(o,dpi=120)
    sns.heatmap(data=torch.max(lambdamax_layers[:,0:2,:],dim=1)[0],
                vmin=1e11,
                vmax=1e16,
                cmap=plt.get_cmap('Blues')
        )
    plt.title('LAMBDAmax_IN_HEADS_EVERY_LAYERS') 
    plt.savefig('NTKS-0330/NTK-lambdamax-max-%dsteps-%s.png'%(steps,doamin)) 
    o+=1

    for w in range(W):
        plt.figure(o,dpi=120)
        sns.heatmap(data=lambdamax_layers[:,w,:],
                    vmin=1e11,
                    vmax=1e16,
                    cmap=plt.get_cmap('Blues')
            )
        plt.title('LAMBDAmax_IN_HEADS_EVERY_LAYERS') 
        plt.savefig('NTKS-0330/NTK-lambdamax-%dsteps-%dW-%s.png'%(steps,w,doamin)) 
        o+=1
    plt.figure(o,dpi=120)
    sns.heatmap(data=torch.max(longtailed_NTK[:,0:2,:], dim=1)[0],
                vmin=-1,
                vmax=1,
                cmap=plt.get_cmap('Blues')
        )
    plt.title('PROJECTIONS_IN_HEADS_EVERY_LAYERS') 
    plt.savefig('NTKS-0330/NTK-projection-max-%dsteps-%s.png'%(steps,doamin)) 
    o+=1
    for w in range(W):
        plt.figure(o,dpi=120)
        sns.heatmap(data=longtailed_NTK[:,w,:],
                    vmin=-1,
                    vmax=1,
                    cmap=plt.get_cmap('Blues')
            )
        plt.title('PROJECTIONS_IN_HEADS_EVERY_LAYERS') 
        plt.savefig('NTKS-0330/NTK-projection-%dsteps-%dW-%s.png'%(steps,w,doamin)) 
        o+=1
    plt.figure(o,dpi=120)
    sns.heatmap(data=torch.max(longtailed_sim[:,0:2,:],dim=1)[0],
                vmin=-1,
                vmax=1,
                cmap=plt.get_cmap('Blues')
        )
    plt.title('SIMILARITY_IN_HEADS_EVERY_LAYERS') 
    plt.savefig('NTKS-0330/NTK-similarity-max-%dsteps-%s.png'%(steps,doamin)) 
    o+=1
    for w in range(W):
        plt.figure(o,dpi=120)
        sns.heatmap(data=longtailed_sim[:,w,:],
                    vmin=-1,
                    vmax=1,
                    cmap=plt.get_cmap('Blues')
            )
        plt.title('SIMILARITY_IN_HEADS_EVERY_LAYERS') 
        plt.savefig('NTKS-0330/NTK-similarity-%dsteps-%dW-%s.png'%(steps,w,doamin)) 
        o+=1
    print(steps,"finished")
    return NTKs, NTK,lambdamax_layers,o


# https://github.com/pnnl/torchntk/blob/master/torchntk/autograd/autograd_ntk.py
def get_ntk_by_layer_from_yubin(model, loader):

    NTKs = {}

    params_that_need_grad = []
    for param in model.parameters():
        if param.requires_grad:
            params_that_need_grad.append(param.requires_grad)
            param.requires_grad = False # first set all gradients to not calculate, time saver
        else:
            params_that_need_grad.append(param.requires_grad)
    
    for index, (name, param) in enumerate(model.named_parameters()):
        if not params_that_need_grad[index]: #if it didnt need a grad, we can skip it.
            continue
        param.requires_grad = True #we only care about this tensors gradients in the loop
        this_grad = []

        for i, batch in enumerate(loader):
            if i >= 1: break

            _, logits = model(**batch)
            for i in range(len(logits)):
                
                logits[i:i+1].backward(torch.ones_like(logits[i:i+1]), create_graph=True)
                this_grad.append(param.grad.detach().reshape(-1).clone())
                model.zero_grad()
                torch.cuda.empty_cache()

        J_layer = torch.stack(this_grad) # [N x P matrix] #this will go against our notation, but I'm not adding

        NTKs[name] = J_layer @ J_layer.T # An extra transpose operation to my code for us to feel better

        param.requires_grad = False
     
    #reset the model object to be how we started this function
    for i, param in enumerate(model.parameters()):
        if params_that_need_grad[i]:
            param.requires_grad = True

    NTK = torch.sum(torch.stack([val for val in NTKs.values()]) ,dim=0)
    return NTKs, NTK