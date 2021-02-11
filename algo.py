import sys
from pprint import pprint

import matplotlib as mpl

# mpl.rcParams["figure.dpi"] = 600
import matplotlib.pyplot as plt
import numpy as np
from tqdm import trange

import seaborn as sns

sns.set()
# np.random.seed(24)

INF = 10


class Problem:
    def __init__(self):
        pass

    def get_start_population(self):
        raise NotImplementedError()

    def fitness_fn(self, individual):
        raise NotImplementedError()


class TSP(Problem):
    def __init__(self):
        super().__init__()
        self.distances = {}

        for i in range(14):
            self.distances[chr(ord("A") + i)] = {}

        #                           A       B         C         D         E         F        G          H        I         J         K          L       M        N
        self.distances["A"] = {
            "A": 0,
            "B": INF,
            "C": INF,
            "D": INF,
            "E": INF,
            "F": INF,
            "G": 0.15,
            "H": INF,
            "I": INF,
            "J": 0.2,
            "K": INF,
            "L": 0.12,
            "M": INF,
            "N": INF,
        }
        self.distances["B"] = {
            "A": INF,
            "B": 0,
            "C": INF,
            "D": INF,
            "E": INF,
            "F": INF,
            "G": INF,
            "H": 0.19,
            "I": 0.4,
            "J": INF,
            "K": INF,
            "L": INF,
            "M": INF,
            "N": 0.13,
        }
        self.distances["C"] = {
            "A": INF,
            "B": INF,
            "C": 0,
            "D": 0.6,
            "E": 0.22,
            "F": 0.4,
            "G": INF,
            "H": INF,
            "I": 0.2,
            "J": INF,
            "K": INF,
            "L": INF,
            "M": INF,
            "N": INF,
        }
        self.distances["D"] = {
            "A": INF,
            "B": INF,
            "C": 0.6,
            "D": 0,
            "E": INF,
            "F": 0.21,
            "G": INF,
            "H": INF,
            "I": INF,
            "J": INF,
            "K": 0.3,
            "L": INF,
            "M": INF,
            "N": INF,
        }
        self.distances["E"] = {
            "A": INF,
            "B": INF,
            "C": 0.22,
            "D": INF,
            "E": 0,
            "F": INF,
            "G": INF,
            "H": INF,
            "I": 0.18,
            "J": INF,
            "K": INF,
            "L": INF,
            "M": INF,
            "N": INF,
        }
        self.distances["F"] = {
            "A": INF,
            "B": INF,
            "C": 0.4,
            "D": 0.21,
            "E": INF,
            "F": 0,
            "G": INF,
            "H": INF,
            "I": INF,
            "J": INF,
            "K": 0.37,
            "L": 0.6,
            "M": 0.26,
            "N": 0.9,
        }
        self.distances["G"] = {
            "A": 0.15,
            "B": INF,
            "C": INF,
            "D": INF,
            "E": INF,
            "F": INF,
            "G": 0,
            "H": INF,
            "I": INF,
            "J": INF,
            "K": 0.55,
            "L": 0.18,
            "M": INF,
            "N": INF,
        }
        self.distances["H"] = {
            "A": INF,
            "B": 0.19,
            "C": INF,
            "D": INF,
            "E": INF,
            "F": INF,
            "G": INF,
            "H": 0,
            "I": INF,
            "J": 0.56,
            "K": INF,
            "L": INF,
            "M": INF,
            "N": 0.17,
        }
        self.distances["I"] = {
            "A": INF,
            "B": 0.4,
            "C": 0.2,
            "D": INF,
            "E": 0.18,
            "F": INF,
            "G": INF,
            "H": INF,
            "I": 0,
            "J": INF,
            "K": INF,
            "L": INF,
            "M": INF,
            "N": 0.6,
        }
        self.distances["J"] = {
            "A": 0.2,
            "B": INF,
            "C": INF,
            "D": INF,
            "E": INF,
            "F": INF,
            "G": INF,
            "H": 0.56,
            "I": INF,
            "J": 0,
            "K": INF,
            "L": 0.16,
            "M": INF,
            "N": 0.5,
        }
        self.distances["K"] = {
            "A": INF,
            "B": INF,
            "C": INF,
            "D": 0.3,
            "E": INF,
            "F": 0.37,
            "G": 0.55,
            "H": INF,
            "I": INF,
            "J": INF,
            "K": 0,
            "L": INF,
            "M": 0.24,
            "N": INF,
        }
        self.distances["L"] = {
            "A": 0.12,
            "B": INF,
            "C": INF,
            "D": INF,
            "E": INF,
            "F": 0.6,
            "G": 0.18,
            "H": INF,
            "I": INF,
            "J": 0.16,
            "K": INF,
            "L": 0,
            "M": 0.4,
            "N": INF,
        }
        self.distances["M"] = {
            "A": INF,
            "B": INF,
            "C": INF,
            "D": INF,
            "E": INF,
            "F": 0.26,
            "G": INF,
            "H": INF,
            "I": INF,
            "J": INF,
            "K": 0.24,
            "L": 0.4,
            "M": 0,
            "N": INF,
        }
        self.distances["N"] = {
            "A": INF,
            "B": 0.13,
            "C": INF,
            "D": INF,
            "E": INF,
            "F": 0.9,
            "G": INF,
            "H": 0.17,
            "I": 0.6,
            "J": 0.5,
            "K": INF,
            "L": INF,
            "M": INF,
            "N": 0,
        }

    def get_start_population(self):
        path = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M", "N"]
        list_population = [path for i in range(20)]
        return np.stack(list_population, axis=0)

    def fitness_fn(self, config):
        res = 0
        for i in range(len(config) - 1):
            res += self.distances[config[i]][config[i + 1]]

        return 1500 - res


class NQueens(Problem):
    def __init__(self):
        super().__init__()

    def get_start_population(self):
        random_val = np.random.randint(8)
        list_population = [np.ones(8, dtype=int) * random_val for i in range(20)]
        return np.stack(list_population, axis=0)

    def fitness_fn(self, board):
        board=np.array(board)
        n = board.shape[-1]
        row_frequency = [0] * n
        main_diag_frequency = [0] * (2 * n)
        secondary_diag_frequency = [0] * (2 * n)

        for i in range(n):
            row_frequency[board[i]] += 1
            main_diag_frequency[board[i] + i] += 1
            secondary_diag_frequency[n - board[i] + i] += 1

        conflicts = 0
        # formula: (N * (N - 1)) / 2
        for i in range(2 * n):
            if i < n:
                conflicts += (row_frequency[i] * (row_frequency[i] - 1)) / 2
            conflicts += (main_diag_frequency[i] * (main_diag_frequency[i] - 1)) / 2
            conflicts += (
                secondary_diag_frequency[i] * (secondary_diag_frequency[i] - 1)
            ) / 2
        return 1 + (28 - int(conflicts))


class GeneticAlgo:
    def __init__(self, problem):
        self.problem = problem
        self.population = self.problem.get_start_population()
        self.fitnesses = [np.max(
                    np.array(
                        [self.problem.fitness_fn(member) for member in self.population]
                    )
                )]

    def selector(self, population, fitness_fn, size=1):
        weights = np.array([fitness_fn(board) for board in population])
        probs = [weight / sum(weights) for weight in weights]
        # probs = [np.exp(weight)/np.sum(np.exp(weights)) for weight in weights]
        # probs = [weight*weight for weight in weights]
        # probs = [prob/sum(probs) for prob in probs]
        indices = np.random.choice(
            [i for i in range(len(population))], size=size, p=probs
        )
        population = np.array(population)
        return population[indices]

    def reproduce(self, x, y):
        slice_idx = np.random.randint(8)
        new_guy = np.array(x[: int(slice_idx)])
        return np.append(new_guy, np.array(y[int(slice_idx) :]))

    def mutate(self, x, force=False):
        if np.random.rand() < 0.1 or force:
            x[np.random.randint(8)] = np.random.randint(8)
        return x

    def train(self, num_iterations=100):
        with trange(num_iterations) as t:
            for epoch in t:
                new_population = []
                for i in range(len(self.population)):
                    x = self.selector(self.population, self.problem.fitness_fn)
                    y = self.selector(self.population, self.problem.fitness_fn)
                    child = self.reproduce(x, y)
                    child = self.mutate(child)
                    new_population.append(child)
                self.population = new_population
                best_individual = np.argmax(
                    np.array([self.problem.fitness_fn(member) for member in new_population])
                )
                self.fitnesses.append(
                    np.max(
                        np.array(
                            [self.problem.fitness_fn(member) for member in new_population]
                        )
                    )
                )
                t.set_postfix(maxF=self.fitnesses[-1])
                if self.fitnesses[-1] == 29:
                    # break
                    pass
                if epoch == 400:
                    # pprint(self.population)
                    pass
                # t.set_postfix(fitness=fitnesses[-1], gen=epoch)

    def plot_fitnesses(self):
        sns.set_palette("colorblind")
        sns.set_context("paper")
        plt.plot(self.fitnesses)
        plt.title("Genetic Algorithm on nQueens")
        plt.show()


class FastGeneticAlgo(GeneticAlgo):
    def __init__(self, problem):
        super().__init__(problem)
        
    def reproduce(self, x, y):
        candidates=[]
        for slice_idx in range(8):
            x=np.array(x)
            y=np.array(y)
            x = list(x.flatten())
            y = list(y.flatten())
            candidate1 = x[:slice_idx]+y[slice_idx:]
            candidate2 = x[:8-slice_idx]+y[8-slice_idx:]
            candidates.append(candidate1)
            candidates.append(candidate2)
        candidates=np.array(candidates)
        res = self.greedy_selector(np.array(candidates), self.problem.fitness_fn)
        return res.flatten()

    def greedy_selector(self, population, fitness_fn):
        weights = [fitness_fn(board) for board in population]
        idx = np.argmax(weights)
        return population[idx]
    
    def selector(self, population, fitness_fn, size=1):
        weights = [fitness_fn(board) for board in population]
        probs = [np.exp(weight) / np.sum(np.exp(weights)) for weight in weights]
        indices = np.random.choice(
            [i for i in range(len(population))], size=size, p=probs
        )
        population = np.array(population)
        return population[indices]
    
    def train(self, num_iterations=100):
        with trange(num_iterations) as t:
            for epoch in t:
                new_population = []
                for i in range(len(self.population)):
                    x = self.selector(self.population, self.problem.fitness_fn)
                    y = self.selector(self.population, self.problem.fitness_fn)
                    child = self.reproduce(x, y)
                    child = self.mutate(child, force=epoch>200 and self.fitnesses[-1]==self.fitnesses[-10] and self.fitnesses[-1]!=29)
                    new_population.append(child)
                self.population = new_population
                best_individual = np.argmax(
                    np.array([self.problem.fitness_fn(member) for member in new_population])
                )
                self.fitnesses.append(
                    np.max(
                        np.array(
                            [self.problem.fitness_fn(member) for member in new_population]
                        )
                    )
                )
                t.set_postfix(maxF=self.fitnesses[-1])


class TSPAlgo(GeneticAlgo):
    def __init__(self, problem):
        super().__init__(problem)

    def mutate(self, x):
        x=np.array(x)
        idx1 = np.random.randint(x.shape[-1])
        idx2 = np.random.randint(x.shape[-1])
        temp = x[idx1]
        x[idx1] = x[idx2]
        x[idx2] = temp
        return x

    def reproduce(self, x, y):
        idx1 = np.random.randint(x.shape[-1])
        idx2 = idx1 + 2
        res = np.array([chr(ord("A")+i) for i in range(x.shape[-1])])
        # print(idx1, idx2)
        # print(res[idx1:idx2])
        # print(x.flatten())
        print(f"idx1: {idx1}, idx2: {idx2}")
        res[idx1:idx2] = x.flatten()[idx1:idx2]
        i = 0
        for e in y:
            if e not in res and i not in range(idx1, idx2):
                res[i] = e
                i = (i + 1 + (x.shape[-1])) % (x.shape[-1])

        return res

    def selector(self, population, fitness_fn, size=1):
        weights = [fitness_fn(board) for board in population]
        # return population[np.argmax(weights)]
        probs = [np.exp(weight) / np.sum(np.exp(weights)) for weight in weights]
        indices = np.random.choice(
            [i for i in range(len(population))], size=size, p=probs
        )
        population = np.array(population)
        return population[indices]


if __name__ == "__main__":

    avg_fitnesses_fast = []
    avg_fitnesses_slow = []
    with trange(100) as t:
        for i in t:
            n_queens = NQueens()
            my_genetic_algo = GeneticAlgo(n_queens)
            fast_genetic_algo = FastGeneticAlgo(n_queens)
            my_genetic_algo.train(int(sys.argv[1]))
            fast_genetic_algo.train(int(sys.argv[1]))
            avg_fitnesses_fast.append((fast_genetic_algo.fitnesses))
            avg_fitnesses_slow.append(my_genetic_algo.fitnesses)
            t.set_postfix(maxFast=max(fast_genetic_algo.fitnesses), maxSlow=max(my_genetic_algo.fitnesses), gFast=len(fast_genetic_algo.fitnesses), gSlow=len(my_genetic_algo.fitnesses))
            if (i+1)%2 == 0:
                print(len(avg_fitnesses_fast), len(avg_fitnesses_fast[0]))
                sns.set_palette("colorblind")
                sns.set_context("paper")
                plt.suptitle("Genetic Algorithm on NQueens", fontsize='x-large')
                plt.title(f"Averaged over {i+1} runs")
                plt.xlabel("#Generation")
                plt.yticks(np.arange(0, 31, step=2))
                plt.ylabel("Max Fitness of the Population")
                plt.plot(np.mean(avg_fitnesses_fast, axis=0), label="Optimized Algorithm")
                plt.plot(np.mean(avg_fitnesses_slow, axis=0), label="Original Algorithm")
                plt.legend(loc="best")
                plt.savefig(f'./Plots/CompGen2Plot/CompGen2Plot{i+1}.png')
                plt.clf()
            # if (i+1)%10 == 0:
            #     fig, ax = plt.subif(self.problem.fitness_fn(candidate1)>self.problem.fitness_fn(candidate2)):
            #     return candidate1
            # else:
            #     return candidate2plots(1, 1)
            #     ax.hist(num_counts)
            #     ax.set_title("Histogram of result")

            #     ax.set_xlabel("Generations")
            #     ax.set_ylabel("Number of times")
            #     plt.savefig(f"./Plots/FastHistMax_{i}.png")

    # tsp_algo = FastGeneticAlgo(queens)
    # tsp_algo.train(int(sys.argv[1]))
    # tsp_algo.plot_fitnesses()
    
    # n_queens = NQueens()
    # fast_algo = FastGeneticAlgo(n_queens)
    # fast_algo.train(int(sys.argv[1]))
    # my_genetic_algo = GeneticAlgo(n_queens)
    # my_genetic_algo.train(int(sys.argv[1]))
    # sns.set_context("paper")
    # plt.plot(my_genetic_algo.fitnesses, label="Slowww")
    # plt.plot(fast_algo.fitnesses, label="FASTTTTT")
    # plt.legend(loc="best")
    # # my_genetic_algo.train()
    # plt.title("Genetic Algorithm on nQueens")
    # # plt.show()
