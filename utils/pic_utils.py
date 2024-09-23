import torch
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA


def scatter_plot(x, y, xlabel, ylabel, title):
    plt.figure(0)
    plt.scatter(x, y, alpha=0.5, c='k')
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True)
    plt.savefig(title + '.png')
    
    
def pca_scatter(data1, data2, title):
    """Draw a 2-dim pca scatter plot to see the distribution of data1 & data2

    Args:
        data1 (tensor): [batch_size, seq_len, hidden_dim]
        data2 (tensor): [batch_size, seq_len, hidden_dim]
        title (str): the title of pic & the name of saved pic
    """
    data = torch.cat((data1, data2), dim=0)
    data = data.mean(dim=1)
    data = data.cpu().numpy()
    
    pca = PCA(n_components=2)
    data_pca = pca.fit_transform(data)
    plt.figure()
    plt.scatter(data_pca[:len(data1), 0], data_pca[:len(data1), 1], alpha=0.5, c='b')
    plt.scatter(data_pca[len(data1):, 0], data_pca[len(data1):, 1], alpha=0.5, c='r')
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.title('PCA Result')
    plt.grid(True)
    plt.savefig(title + '.png')