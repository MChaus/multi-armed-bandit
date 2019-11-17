import numpy as np
from abc import ABC, abstractmethod

class Strategy(ABC):
    @abstractmethod
    def __init__(self):
        pass
    
    @abstractmethod
    def estimate_values(self, preferences, number_of_actions, indx, reward):
        pass
        

class SampleAverages(Strategy):
    def __init__(self):
        pass
    
    def estimate_values(self, preferences, number_of_actions, indx, reward):
        preferences[indx] += 1 / number_of_actions[indx] * (reward - preferences[indx])
        return preferences

   
class ConstantStepSize(Strategy):
    def __init__(self, alpha=0.1):
        self.alpha = alpha

    def estimate_values(self, preferences, number_of_actions, indx, reward):
        preferences[indx] += self.alpha * (reward - preferences[indx])
        return preferences


class UCB(Strategy):
    def __init__(self, c=2):
        self.c = c
        self.time = 1
        self.values = None
        
    def _init_values(self, length):
        if self.values is None:
            self.values = np.zeros(length)
    
    def estimate_values(self, preferences, number_of_actions, indx, reward):
        if self.values is None:
            self._init_values(len(preferences))
        self.time += 1
        self.values[indx] += 1 / number_of_actions[indx] * (reward - self.values[indx])
        return self.values + self.c * (np.log(self.time) / number_of_actions) ** 0.5
        
 
class GradientBandit(Strategy):
    def __init__(self, alpha=0.1):
        self.alpha = alpha
        self.mean_reward = 0
        self.time = 0
        self.probs = None
        
    def _init_probs(self, length):
        if self.probs is None:
            self.probs = np.ones(length) / length
    
    def _derive_probs(self, preferences):
        self.probs = np.exp(preferences)
        self.probs /= sum(self.probs)
    
    def estimate_values(self, preferences, number_of_actions, indx, reward):
        if self.probs is None:
            self._init_probs(len(preferences))
        self.time += 1
        self.mean_reward += 1 / self.time * (reward - self.mean_reward)
        preferences -= self.alpha * (reward - self.mean_reward) * self.probs
        preferences[indx] += self.alpha * (reward - self.mean_reward)
        self._derive_probs(preferences)
        return preferences