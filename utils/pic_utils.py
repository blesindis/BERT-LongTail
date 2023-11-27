import matplotlib.pyplot as plt


def line_chart(x, y, xlabel, ylabel, title):
    plt.figure(0)
    plt.scatter(x, y, alpha=0.5, c='k')
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True)
    plt.savefig(title + '.png')