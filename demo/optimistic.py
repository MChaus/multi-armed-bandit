from src import bandit_utils
from src import strategies

def init_players(n_steps):
    players = list()
    bandit = bandit_utils.Bandit(n=10)
    players.append(bandit_utils.BanditPlayer(bandit, eps=0, n_steps=n_steps, 
                                             strategy=strategies.ConstantStepSize(alpha=0.1),
                                             optimist=True, baseline=5, 
                                             name=r'$Optimistic$ $greedy, Q=5,'
                                                  r'\alpha=0.1, \varepsilon=0$'))
    players.append(bandit_utils.BanditPlayer(bandit, eps=0.1, n_steps=n_steps, 
                                             strategy=strategies.ConstantStepSize(alpha=0.1),
                                             name=r'$Realistic, \varepsilon-greedy,'
                                                  r'Q=0, \varepsilon=0.1$'))
    return players


def main():
    n_steps = 1000
    n_epochs = 2000
    
    players = init_players(n_steps)
    benchmark = bandit_utils.Benchmark(players, n_epochs, r'$Optimistic$',
                                       print_log=False, log_time=100000)
    benchmark.run()
    
if __name__ == '__main__':
    main()
