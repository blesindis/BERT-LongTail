import torch


def DBI(data1, data2):
    c1, c2 = torch.mean(data1, dim=0), torch.mean(data2, dim=0)    
    sigma1, sigma2 = torch.mean(torch.norm(data1 - c1, dim=1, p=2), dim=0), torch.mean(torch.norm(data2 - c2, dim=1, p=2), dim=0)
    dbi = (sigma1 + sigma2) / torch.norm(c1-c2, p=2)
    # print(c1, c2, sigma1, sigma2)
    return dbi