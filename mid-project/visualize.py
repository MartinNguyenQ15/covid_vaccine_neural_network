import seaborn as sns
import matplotlib.pyplot as plt

def show_heatmap(df, title, center="dark", hneg=250, hpos=354, s=80, l=60, corr_method="pearson"):
    ax = plt.axes()
    corr = df.corr(method=corr_method)
    cmap = sns.diverging_palette(hneg, hpos, s, l, center=center, as_cmap=True)
    sns.heatmap(corr, vmax=1, vmin=-.5, cmap=cmap, square=True, linewidths=.2)
    ax.set_title(title)
    plt.show()

# WIP
def show_histogram(df, title, xlabels=[], rows=1, cols=1):
    print(xlabels)
    colors = ['r', 'g', 'b']
    current_label = 0
    fig, axes = plt.subplots(rows, cols, figsize=(10, 6), sharey=True)
    if (rows == 1 and cols == 1):
        sns.histplot(df, ax=axes, x=xlabels[0], kde=True, color='r')
    else:
        for r in range(rows):
            for c in range(cols):
                sns.histplot(df,ax=axes[r, c], x=xlabels[current_label], kde=True, color=colors[(r + c) % len(colors)])
    ax = plt.axes()
    ax.set_title(title)
    plt.show()