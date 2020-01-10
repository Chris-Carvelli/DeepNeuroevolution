import torch
import pandas as pd
import matplotlib.pyplot as plt
from src.utils.OptimalTiling import count_trainable
import seaborn as sns
sns.set()

SOLVED = {
    'BipedalWalker-v2': 200,
    'LunarLanderContinuous-v2': 200,
    'FrostbiteDeterministic-v4': None,
    'CarRacing-v0': 900,
}

# hnn_models = ['hnn-world-models', 'hnn_0.125']
hnn_models = ['hnn']
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
    'ann_compression',
    'hnn'
]

X_MAX = 1000
folders = ['atari']
# style = 'dark_background'
style = None


if style is not None:
    plt.style.use('dark_background')


def plot(y, env=None):
    # fig = plt.figure()
    palette = (
        sns.color_palette('Oranges_r', data[data['kind'] == 'pnn']['model'].drop_duplicates().count()) +
        sns.color_palette('Blues_r', data[data['kind'] == 'hnn']['model'].drop_duplicates().count())
    )
    sns.lineplot('epoch', y, 'model', data=data[data['epoch'] < X_MAX], palette=palette)

    if style is not None:
        fig.patch.set_facecolor('None')
        fig.patch.set_alpha(0)
    plt.title(env)
    # plt.savefig(f'plots/{env}_{y}.png')
    plt.show()


def plot_grid(y):
    grid = sns.FacetGrid(data[gen_filter & model_filter], col='env', sharex=False, sharey=False)
    grid.map(sns.lineplot, 'epoch', y, 'model')
    for i, ax in enumerate(grid.axes.flatten()):
        if style is not None:
            ax.patch.set_alpha(0)
        if SOLVED[grid.col_names[i]] is not None:
            ax.hlines(SOLVED[grid.col_names[i]], 0, X_MAX, colors='red', linestyles='--')
    grid.fig.patch.set_alpha(0)

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

            # TMP
            try:
                state_dict = torch.load(f'results/{folder}/{run}/best.p')
            except:
                from examples.racing.models.HyperNN import HyperNN
                from examples.racing.models.worldmodels.PolicyNN import PolicyNN
                import gym
                env = gym.make('CarRacing-v0')
                tmp = HyperNN(env.observation_space, env.action_space, PolicyNN, 1024)
                state_dict = tmp.state_dict()
            filter = 'hnn' if run is 'hnn' else None
            param_count = count_trainable(state_dict, filter)

            df['model'] = run_name
            df['best'] = best
            df['Params'] = param_count
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

gen_filter = data['epoch'] < X_MAX
model_filter = data['model'] != 'ga-world-models'
plot('elite_avg', data['env'][0])
plot('best', data['env'][0])
plot('Elite average windowed (50)', data['env'][0])

summary = data[['model', 'Params']].groupby('model').mean().reset_index().sort_values('Params')
print(summary)

