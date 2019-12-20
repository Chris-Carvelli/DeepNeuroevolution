from tqdm import tqdm
from time import sleep

for i in tqdm(range(4), desc='1st loop', leave=True, position=0, ncols=100):
    for j in tqdm(range(5), desc='2nd loop', leave=False, position=1, ncols=100):
        for k in tqdm(range(50), desc='3nd loop', leave=False, position=2, ncols=100):
            sleep(0.01)
