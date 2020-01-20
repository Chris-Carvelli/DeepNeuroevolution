import math
import torch
import pandas as pd
import matplotlib.pyplot as plt
from src.utils.OptimalTiling import count_buffers, count_params, count_trainable
import seaborn as sns
sns.set()

X = 'evals'
X_LIM = (0, 1e9)
Y_LIM = None
GRID = True

folders = [
    'asteroids',
    'Amidar',
    # 'Assault',
    # 'Asterix',
    # 'Atlantis',
    # 'Enduro',
    'Gravitar',
    # 'Kangaroo',
    # 'Seaquest',
    # 'Zaxxon'
    'atari'
]
# style = 'dark_background'
style = None
# hnn_models = ['hnn-world-models', 'hnn_0.125']
hnn_models = ['hnn']

SOLVED = {
    'BipedalWalker-v2': 200,
    'LunarLanderContinuous-v2': 200,
    'FrostbiteDeterministic-v4': 4000,
    'AmidarDeterministic-v4': 250,
    'AssaultDeterministic-v4': 500,
    'AsterixDeterministic-v4': 1500,
    'AtlantisDeterministic-v4': 25000,
    'EnduroDeterministic-v4': 50,
    'GravitarDeterministic-v4': 500,
    'KangarooDeterministic-v4': 1500,
    'SeaquestDeterministic-v4': 500,
    'SkiingDeterministic-v4': -6500,
    'VentureDeterministic-v4': 250,
    'ZaxxonDeterministic-v4': 4000,
    'CarRacing-v0': 900,
}

runs = [
    # # Racing - cnn
    # 'cnn',
    # 'cnn_compressed',
    # 'hcnn',
    # # Racing - world models
    # # 'ga-world-models',
    # 'ann_0.125',
    # 'hnn_0.125',
    # 'hnn-world-models',
    # Atari
    'ann',
    'ann_0.5',
    'ann_0.25',
    'ann_0.125',
    'hnn'
]


if style is not None:
    plt.style.use('dark_background')


def plot(y, env=None):
    palette = (
        sns.color_palette('Oranges_r', data[data['kind'] == 'pnn']['model'].drop_duplicates().count()) +
        sns.color_palette('Blues_r', data[data['kind'] == 'hnn']['model'].drop_duplicates().count())
    )
    if 'Params' in data.columns:
        fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(12, 6))
        ax1.set_yscale('log')
        sns.barplot(x='model', y='Params', data=data, palette=palette, ax=ax1)
        if style is not None:
            fig.patch.set_facecolor('None')
            fig.patch.set_alpha(0)

        fig.suptitle(env, size=16)
        fig.subplots_adjust(top=.9)
        fig.tight_layout()
    else:
        ax0 = None

    ax0 = sns.lineplot(X, y, 'model', data=data[data[X] < X_LIM[1]], palette=palette, ax=ax0)
    ax0.set_xlim(X_LIM)
    ax0.set_ylim(Y_LIM)

    # plt.savefig(f'plots/{env}_{y}.png')
    plt.show()


def plot_grid(y):
    palette = (
            sns.color_palette('Oranges_r', data[data['kind'] == 'pnn']['model'].drop_duplicates().count()) +
            sns.color_palette('Blues_r', data[data['kind'] == 'hnn']['model'].drop_duplicates().count())
    )
    plt.figure(figsize=(24, 24))
    grid = sns.FacetGrid(data[data['evals'] < X_LIM[1]], col='env', sharex=False, sharey=False, col_wrap=int(math.sqrt(len(folders))))
    grid.map(sns.lineplot, X, y, 'model', palette=palette)
    for i, ax in enumerate(grid.axes.flatten()):
        if grid.col_names[i] in SOLVED:
            ax.hlines(SOLVED[grid.col_names[i]], 0, X_LIM[1], colors='red', linestyles='--')
    # grid.fig.patch.set_alpha(0)
    # plt.legend()
    plt.savefig(f'{y}.png')
    plt.show()


data = pd.DataFrame()
for folder in folders:
    for run in runs:
        try:
            df = pd.read_csv(f'results/{folder}/{run}/res.csv')
            best = df['elite_avg'].copy()
            for i, row in df.iterrows():
                if i > 0:
                    best[i] = max(best[i - 1], df['elite_avg'][i])
            # TMP
            if 'env' not in df.columns:
                df['env'] = 'FrostbiteDeterministic-v4' if 'atari' in folder else 'CarRacing-v0'
            run_name = run if '_compress' not in run else run.split('_')[0] + '_crippled'

            try:
                filter = 'hnn' if run is 'hnn' else None
                state_dict = torch.load(f'results/{folder}/{run}/best.p')
                df['Params'] = count_trainable(state_dict, filter)
            except:
                pass

            df['model'] = run_name
            df['best'] = best
            # df['param_size'] = count_params(state_dict, filter)
            # df['z_size'] = count_buffers(state_dict, filter)
            df['run'] = run
            df['kind'] = 'hnn' if run_name in hnn_models else 'pnn'
            data = data.append(df, ignore_index=True)
        except FileNotFoundError as e:
            print(f'{run} not found')

data = data.rename(columns={'Unnamed: 0': 'epoch'})
data['Elite average windowed (50)'] = data.groupby(['env', 'model'])['elite_avg'].transform(
    lambda x: x.rolling(50).mean()
)
data['Elite std windowed (50)'] = data.groupby(['env', 'model'])['elite_avg'].transform(
    lambda x: x.rolling(50).std()
)

gen_filter = data['epoch'] < X_LIM[1]
model_filter = data['model'] != 'ga-world-models'

if GRID:
    plot_grid('elite_avg')
    plot_grid('best')
    plot_grid('Elite average windowed (50)')
else:
    plot('elite_avg', data['env'][0])
    plot('best', data['env'][0])
    plot('median', data['env'][0])
    plot('mean', data['env'][0])
    plot('Elite average windowed (50)', data['env'][0])

    if 'Params' in data.columns:
        summary = data[['model', 'Params']].groupby('model').mean().reset_index().sort_values('Params')
    print(summary)


