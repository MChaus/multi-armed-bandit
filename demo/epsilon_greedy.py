from src import bandit_utils
from src import strategies

def init_players(n_steps):
    players = list()
    bandit = bandit_utils.Bandit(n=10, incr_dev=0)
    players.append(bandit_utils.BanditPlayer(bandit, eps=0, n_steps=n_steps, 
                                             strategy=strategies.SampleAverages(),
                                             name=r'$\varepsilon=0 (greedy)$'))
    players.append(bandit_utils.BanditPlayer(bandit, eps=0.01, n_steps=n_steps, 
                                             strategy=strategies.SampleAverages(),
                                             name=r'$\varepsilon=0.01$'))
    players.append(bandit_utils.BanditPlayer(bandit, eps=0.1, n_steps=n_steps, 
                                             strategy=strategies.SampleAverages(),
                                             name=r'$\varepsilon=0.1$'))
    return players

def main():
    n_epochs = 2000
    n_steps = 1000
    players = init_players(n_steps)
    benchmark = bandit_utils.Benchmark(players, n_epochs, r'$\varepsilon-greedy$',
                                       print_log=False, log_time=100000)
    benchmark.run()
    
if __name__ == '__main__':
    main()
