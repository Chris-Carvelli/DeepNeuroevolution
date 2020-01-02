import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()


def plot(y):
    sns.lineplot('epoch', y, 'model', data=data[gen_filter & model_filter])
    plt.show()


runs = [
    'cnn',
    'hcnn',
    'ga-world-models',
    'hnn-world-models',
    'ga-world-models_original'
]

data = pd.DataFrame()
for run in runs:
    df = pd.read_csv(f'results/{run}/res.csv')
    best = df['elite_avg'].copy()
    for i, row in df.iterrows():
        if i > 0:
            best[i] = max(best[i - 1], df['elite_avg'][i])
    df['model'] = run
    df['best'] = best
    data = data.append(df, ignore_index=True)

data = data.rename(columns={'Unnamed: 0': 'epoch'})
data['Elite average windowed (50)'] = data.groupby('model')['elite_avg'].transform(
    lambda x: x.rolling(50).mean()
)

gen_filter = data['epoch'] < 1000
model_filter = data['model'] != 'ga-world-models'
plot('elite_avg')
plot('best')

gen_filter = data['epoch'] < 200
model_filter = data['model'] != 'ga-world-models_original'
plot('elite_avg')
plot('best')
plot('Elite average windowed (50)')
