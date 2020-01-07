import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()

SOLVED = {
    'BipedalWalker-v2': 200,
    'LunarLanderContinuous-v2': 200,
    'FrostbiteDeterministic-v4': None,
    'CarRacing-v0': 900,
}
X_MAX = 1000
folders = ['walker', 'lander', 'atari', 'racing']
style = 'dark_background'


if style is not None:
    plt.style.use('dark_background')


def plot(y, env=None):
    fig = plt.figure()
    sns.lineplot('epoch', y, 'model')
    fig.patch.set_facecolor('None')
    fig.patch.set_alpha(0)
    plt.title(env)
    plt.savefig(f'plots/{env}_{y}.png')


def plot_grid(y):
    grid = sns.FacetGrid(data[gen_filter & model_filter], col='env', sharex=False, sharey=False)
    grid.map(sns.lineplot, 'epoch', y, 'model')
    for i, ax in enumerate(grid.axes.flatten()):
        ax.patch.set_alpha(0)
        if SOLVED[grid.col_names[i]] is not None:
            ax.hlines(SOLVED[grid.col_names[i]], 0, X_MAX, colors='red', linestyles='--')
    grid.fig.patch.set_alpha(0)

    plt.savefig(f'{y}.png')
    plt.show()


runs = [
    'cnn',
    'hcnn',
    'ga-world-models',
    'ga-world-models_original',
    'hnn-world-models',
    'ann',
    'hnn'
]

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
            df['model'] = run
            df['best'] = best
            data = data.append(df, ignore_index=True)
        except FileNotFoundError as e:
            print(f'{run} not found')

data = data.rename(columns={'Unnamed: 0': 'epoch'})
data['Elite average windowed (50)'] = data.groupby(['env', 'model'])['elite_avg'].transform(
    lambda x: x.rolling(50).mean()
)

gen_filter = data['epoch'] < X_MAX
model_filter = data['model'] != 'ga-world-models'
plot('elite_avg', data['env'][0])
plot('best', data['env'][0])
plot('Elite average windowed (50)', data['env'][0])

# gen_filter = data['epoch'] < 200
# model_filter = data['model'] != 'ga-world-models_original'
# plot('elite_avg')
# plot('best')
# plot('Elite average windowed (50)')
