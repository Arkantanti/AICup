import matplotlib.pyplot as plt
import seaborn as sns


def plot_bird_feature(df, column_name, title=None, savefig=False, save_path=None):
    # Set the overall style to dark
    plt.style.use('dark_background')

    # Custom color palette matching the specific colors in your reference image
    # Blue, Orange, Green, Red, Purple, Brown, Pink, Grey, Yellow
    custom_colors = [
        "#0044ff", "#ff7700", "#11cc33", "#ee0000",
        "#8822ff", "#aa5500", "#ff44aa", "#999999", "#ffcc00"
    ]

    # Create figure with the specific dark navy background color
    fig, ax = plt.subplots(figsize=(14, 7), facecolor='#0f111a')
    ax.set_facecolor('#0f111a')

    sns.boxplot(
        data=df,
        x='bird_group',
        y=column_name,
        hue='bird_group',  # Maps colors to groups correctly
        legend=False,  # Removes the redundant legend
        palette=custom_colors,
        showfliers=False,  # Removes the outlier dots
        linewidth=1.5,
        width=0.5,
        ax=ax
    )

    # Formatting Labels and Title
    title_text = f'Boxplot bird groups X {column_name.replace("_", " ")}'
    if title is not None: title_text = f'{title}'
    ax.set_title(title_text, color='white', fontsize=14, pad=20)
    ax.set_xlabel('', fontsize=12)
    ax.set_ylabel('', fontsize=12)

    # Customizing Grid Lines (Horizontal only, faint grey)
    ax.yaxis.grid(True, linestyle='-', which='major', color='#333333', alpha=0.6)
    ax.xaxis.grid(False)

    # Remove outer spines for a clean floating look
    for spine in ['top', 'right', 'left', 'bottom']:
        ax.spines[spine].set_visible(False)

    # Rotate x-labels exactly like the reference image
    plt.xticks(rotation=-45, ha='left', color='white', fontsize=11)
    plt.yticks(color='white')

    plt.tight_layout()

    if savefig:
        plt.savefig(save_path, dpi=300)

    plt.show()
