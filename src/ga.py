import pandas as pd
import torch
import os
import gc
import gym
import copy
import random
import time
import pathos.multiprocessing as mp
import dill

import numpy as np
import sys

torch.set_num_threads(1)


class GA:

    """
    Basic GA implementation.

    The model needs to implement the following interface:
        - evaluate(env, max_eval): evaluate the model in the given environment.
            returns total reward and evaluation used
        - evolve(sigma): evolves the model. Sigma is calculated by the GA
            (usually used as std in an additive Gaussian noise)

    """
    def __init__(self, env_key, population, model_builder,
                 max_generations=20,
                 max_evals=1000*10,
                 max_episode_eval=1000,
                 sigma=0.01,
                 sigma_decay=0.999,
                 min_sigma=0.01,
                 truncation=10,
                 trials=1,
                 elite_trials=0,
                 n_elites=1,
                 reproduce_policy='top',
                 save_folder='results',
                 run_name='hnn'):

        # hyperparams
        self.population = population
        self.model_builder = model_builder
        self.env_key = env_key
        self.max_episode_eval = max_episode_eval
        self.max_evals = max_evals
        self.max_generations = max_generations
        self.sigma = sigma
        self.sigma_decay = sigma_decay
        self.min_sigma = min_sigma
        self.truncation = truncation
        self.trials = trials
        self.elite_trials = elite_trials
        self.n_elites = n_elites
        self.save_folder = save_folder
        self.reproduce_policy = reproduce_policy
        self._reproduce = reproduce_policies[reproduce_policy]

        self.run_name = run_name
        self.log_dir_path = f'./{self.save_folder}/{self.run_name}'
        os.makedirs(os.path.dirname(f'{self.log_dir_path}/'), exist_ok=True)

        # population
        self.scored_parents = None
        self.models = None

        # strategies
        self.termination_strategy = lambda: self.g < self.max_generations and self.evaluations_used < self.max_evals

        # algorithm state
        self.g = 0
        self.evaluations_used = 0
        self.max_score_ever = -sys.float_info.max

        cores = mp.cpu_count()
        self.pool = mp.Pool(cores)
        self.hist = pd.DataFrame()

    def optimize(self):
        """
        Runs a generation of the GA. The result is a tuple
          (median_score, mean_score, max_score, evaluation_used, scored_parents)

        :return: False if the optimization is ended, result otherwise
        """
        if self.termination_strategy():
            if self.models is None:
                self._log(f'{"Res" if self.g > 0 else "S"}tarting {self.run_name}')
                self.models = self._init_models()
                self._log(f'Models init')

            ret = self._evolve_iter()
            self.g += 1
            self._save_checkpoint(ret)
            self._log(f"Gen {self.g}: elite_average: {ret['elite_avg']}")

            return ret
        else:
            self._log('end')
            return False

    def _evolve_iter(self):
        scored_models = self._get_best_models(self.models, self.trials, 'Score Population')
        print('population scored')
        scores = [s for _, s in scored_models]
        median_score = np.median(scores)
        mean_score = np.mean(scores)
        max_score = scored_models[0][1]

        if self.elite_trials > 0:
            scored_candidate_elites = self._get_best_models([m for m, _ in scored_models[:self.truncation]], self.elite_trials, 'Score elite')
            print('elite scored')
        else:
            scored_candidate_elites = scored_models[:self.truncation]

        self._reproduce(self, scored_models, scored_candidate_elites)
        print('population reproduced')

        self.scored_parents = scored_candidate_elites

        return {
            'env': self.env_key,
            'median': median_score,
            'mean': mean_score,
            'max': max_score,
            'elite_avg': scored_candidate_elites[0][1],
            'evals': self.evaluations_used
        }

    def _get_best_models(self, models=None, trials=None, queue_name='Get Best Models'):
        if models is None:
            models = self.models

        if trials is None:
            trials = self.trials

        scored_models, evals = self._score_models(models, trials, queue_name)

        self.evaluations_used += evals
        scored_models = [(m, sum(scores) / trials) for m, scores in scored_models]
        scored_models.sort(key=lambda x: x[1], reverse=True)

        return scored_models

    ###############
    # reproduction
    ###############

    # top n
    def _reproduce_top(self, scored_parents, scored_candidate_elites):
        # Elitism
        self.models = [p for p, _ in scored_parents[:self.n_elites]]
        sigma = max(self.min_sigma, self.sigma * pow(self.sigma_decay, self.g))

        for individual in range(self.population - self.n_elites):
            random_choice = random.choice(scored_parents)
            cpy = copy.deepcopy(random_choice)[0]
            self.models.append(cpy)
            self.models[-1].evolve(sigma)

    # 2 way
    def _reproduce_2way(self, scored_models, scored_candidate_elites):
        new_pop = []
        truncation_size = int(len(scored_models)/2)
        sigma = max(self.min_sigma, self.sigma * pow(self.sigma_decay, self.g))

        # drop worst half
        scored_models = scored_models[:truncation_size]

        while len(new_pop) < truncation_size:
            s1 = random.choice(scored_models)
            s2 = s1

            while s1 == s2:
                s2 = random.choice(scored_models)

            selected_solution = s1[0] if s1[1] > s2[1] else s2[0]
            selected_solution = s1[0] if s1[0] == scored_candidate_elites[0][0] else selected_solution
            selected_solution = s2[0] if s2[0] == scored_candidate_elites[0][0] else selected_solution

            child = copy.deepcopy(selected_solution)
            child.evolve(sigma)
            if child not in new_pop:
                new_pop.append(child)

            del child

        new_pop.extend(list(map(lambda x: x[0], scored_models)))
        # new_pop.extend(new_pop)
        self.models = new_pop

    def _init_models(self):
        self._log('Init models')
        self.pool = mp.Pool(mp.cpu_count())
        if not self.scored_parents:
            env = gym.make(self.env_key)
            ret = [self.model_builder(env.observation_space, env.action_space) for _ in range(self.population)]
            env.close()
            return ret
        else:
            self._reproduce(self, self.scored_parents, None)
            return self.models

    # # serial mode
    # def _score_models(self, models, trials, queue_name):
    #     ret = []
    #     # not pytonic but clear, check performance and memory footprint
    #     evaluation = 0
    #
    #     for m in models:
    #         m_scores = []
    #         for _ in range(trials):
    #             m_scores.append(m.evaluate(self.env, self.max_episode_eval))
    #             evaluation += 1
    #         ret.append((m, m_scores))
    #
    #     return ret

    def _do_eval(self, _models, trials):
        _ret = []
        for m in _models:
            m_scores = []
            for _ in range(trials):
                m_scores.append(m.evaluate(self.env_key, self.max_episode_eval))
            _ret.append((m, m_scores))

        return _ret

    # parallel mode
    def _score_models(self, models, trials, queue_name):
        m_scores = []
        # with mp.Pool(cores) as pool:
        for m in models:
            for _ in range(trials):
                m_scores.append(self.pool.apply_async(m.evaluate, args=(self.env_key, self.max_episode_eval)))
        # self.pool.close()
        # self.pool.join()

        ret = [t.get()[0] for t in m_scores]
        evals = sum([t.get()[1] for t in m_scores])
        ret = np.reshape(ret, (len(models), trials))
        ret = list(zip(models, ret))

        # ret = list(filter(lambda x: len(x) > 0, ret))
        # ret = np.concatenate(ret)
        # ret = np.reshape(ret, (len(models), trials, 2))
        # ret = np.reshape(ret, (len(models), trials, 2))

        gc.collect()
        return ret, evals

    # utils
    def _log(self, s):
        t = time.localtime()
        t_str = time.strftime("%a, %d %b %Y %H:%M:%S", t)
        with open(f'{self.log_dir_path}/ga.log', 'a+') as f:
            f.write(f'[{t_str}] {s}\n')

        print(s)

    def _save_checkpoint(self, hist):
        self.hist = self.hist.append(hist, ignore_index=True)

        self.hist.to_csv(f'{self.log_dir_path}/res.csv')

        with open(f'{self.log_dir_path}/checkpoint.pkl', 'wb+') as fp:
            dill.dump(self, fp)

        if self.scored_parents[0][1] > self.max_score_ever:
            self.max_score_ever = self.scored_parents[0][1]
            torch.save(self.scored_parents[0][0].state_dict(), f'{self.log_dir_path}/best.p')

    # serialization
    def __getstate__(self):
        state = self.__dict__.copy()

        del state['models']
        del state['pool']

        return state

    def __setstate__(self, state):
        self.__dict__.update(state)

        self.models = None


reproduce_policies = {
    'top': GA._reproduce_top,
    '2way': GA._reproduce_2way,
}