'''Provides some essential entities.'''

import copy
import numpy as np
import matplotlib.pyplot as plt

class Arm:
    '''
    Class implements an arm of a multiarmed-bandit.
    
    Parameters
    ----------
    incr_dev : float, optional
        Deviation of normally distributed increment with mean zero, 
        by default 0.
    q : float, optional
        True value of the arm, by default None. If None, value is set
        according to standard normal distribution.
    '''
    
    def __init__(self, incr_dev=0, q=None):
        self.q = q
        self.incr_dev = incr_dev

    @property
    def q(self):
        return self._q

    @q.setter    
    def q(self, value):
        if value is None:
            self._q = np.random.normal()
        else:
            self._q = value

    def action(self):
        ''' Return value of the activated arm ~ N(self.q, 1).'''
        return np.random.normal(self.q)
    
    def reset_q(self):
        '''Increase true value of the arm by random noise ~ N(0, 1) if needed.'''
        if self.incr_dev != 0:
            self.q += np.random.normal(0, self.incr_dev)


class Bandit:
    '''
    Class implements multi-armed bandit.

    Parameters
    ----------
    n : int, optional
        Number of arms, by default 2
    q : list, optional
        True value of each arm, by default None. If None, value is set
        according to standard normal distribution.
    incr_dev : float, optional
        Deviation of arms' normally distributed increment with mean zero, 
        by default 0.
    '''
    
    def __init__(self, n=2, q=None, incr_dev=0):
        self.n = n
        self.q = q
        self.incr_dev = incr_dev
        self._arms = [Arm(self.incr_dev, arm_value) for arm_value in self.q]
    
    @property
    def q(self):
        return self._q

    @q.setter
    def q(self, value):
        if value is None:
            self._q = [None] * self.n
        elif len(value) != self.n:
            raise ValueError('Lenght of arm`s values not equal to the number of'
                             ' arms')
        else:
            self._q = value[:]
        
    @property
    def arms(self):
        return self._arms
    
    def reset(self):
        '''
        Reset all arms to initial values or random values depending on q.
        '''
        self._arms = [Arm(self.incr_dev, arm_value) for arm_value in self.q]
        
    def action(self, indx):
        '''
        Execute one step further.
        
        Parameters
        ----------
        indx : int
            Id of an arm which was activated.
        '''
        for arm in self.arms:
            arm.reset_q()
        return self.arms[indx].action()
    

class BanditPlayer():
    '''
    Class implements user of the multi-armed bandits.
    
    Parameters
    ----------
    bandit : Bandit
        Multi-armed bandit.
    eps : float
        The probability of selecting random action.
    n_steps : int
        Number of actions the player will execute.
    strategy : Strategy
        The followed strategy. 
    optimist : bool, optional
        Whether to add initial baseline to preferences of arms, 
        by default False
    baseline : float, optional
        Basline of the optimistic player, by default 5
    name : str, optional
        Name of the player which will be written on the graphs, 
        by default ''
    '''

    def __init__(self, bandit, eps, n_steps, strategy, optimist=False, 
                 baseline=5, name=''):
        self.eps = eps
        self.n_steps = n_steps
        self.strategy = strategy
        self.optimist = optimist
        self.baseline = baseline
        self._bandit = copy.deepcopy(bandit)
        self.rewards = np.zeros(self.n_steps)
        self._number_of_actions = np.zeros(self.bandit.n)
        self._preferences = np.zeros(self.bandit.n)
        if self.optimist:
            self._preferences += self.baseline
        self.name = name
    
    @property
    def bandit(self):
        return self._bandit

    def run(self):
        ''' 
        Execute all runs of the bandit.
        '''
        for step in range(self.n_steps):
            indx = self.get_arms_id()
            self.rewards[step] = self.use_arm(indx)

    def get_arms_id(self):
        '''
        Pick id of the arm using criteria.
        '''
        indx = 0
        if np.random.binomial(1, self.eps) == 1:
            indx = np.random.randint(0, self.bandit.n)
        else:
            indx = np.argmax(self._preferences)
        return indx

    def use_arm(self, indx):
        '''
        Activate specific arm and recalculate preferences.
        '''
        self._number_of_actions[indx] += 1
        reward = self.bandit.action(indx)
        self._preferences = self.strategy.estimate_values(self._preferences, 
                                                          self._number_of_actions,
                                                          indx, reward)
        return reward
    
    def reset(self):
        '''
        Reset the bandit and all additional fields.
        '''
        self._bandit.reset()
        self.rewards = np.zeros(self.n_steps)
        self._number_of_actions = np.zeros(self.bandit.n)
        self._preferences = np.zeros(self.bandit.n)
        if self.optimist:
            self._preferences += self.baseline
    
class Benchmark:
    '''
    The class implements benchmarking among several players.
    
    Parameters
    ----------
        players : list(BanditPlayer)
            All players which will be tested
        n_epochs : int
            Number of cycles after which the data will be avaraged.
        name : str, optional
            The name of the benchmark which will be places on the graph, 
            by default ''
        print_log : bool, optional
            Whether to print the log or not, by default True
        log_time : int, optional
            A period in epoch after which the graphs will be shown.
    '''
    def __init__(self, players, n_epochs, name='', print_log=True, log_time=100):
        self._n_epochs = n_epochs
        self._players = players
        self._avg_rewards = [np.zeros(player.n_steps) for player in self._players]
        self.name = name
        self.log_time = log_time
        self.print_log = print_log
        
    @property
    def players(self):
        return self._players
        
    def plot_graphs(self, last=False):
        '''
        Plot a representation of obtained data.
        
        Parameters
        ----------
        last : bool, optional
            Whether the graph is a final representation, by default False
        '''
        plt.clf()
        plt.title(self.name)
        for player, avg_reward in zip(self.players, self._avg_rewards):
            plt.plot(avg_reward, label=player.name)
        plt.legend()
        if not last:
            plt.show(block=False)
            plt.pause(0.001) 
        else:
            plt.show()
            
    def run(self):
        '''
        Execute the benchmark.
        '''
        for epoch in range(1, self._n_epochs + 1):
            if self.print_log:
                print('Epoch number: ', epoch)
            for i, player in enumerate(self.players):
                player.run()
                self._avg_rewards[i] += 1 / epoch * (player.rewards - self._avg_rewards[i])
                player.reset()
            if epoch % self.log_time == 0:
                self.plot_graphs()
        self.plot_graphs(True)