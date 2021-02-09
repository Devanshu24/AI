import sys
from pprint import pprint

import matplotlib as mpl

# mpl.rcParams["figure.dpi"] = 600
import matplotlib.pyplot as plt
import numpy as np
from tqdm import trange

import seaborn as sns

sns.set()
np.random.seed(42)

INF = 1e8


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

        return -res


class NQueens(Problem):
    def __init__(self):
        super().__init__()

    def get_start_population(self):
        random_val = np.random.randint(8)
        list_population = [np.ones(8, dtype=int) * random_val for i in range(20)]
        return np.stack(list_population, axis=0)

    def fitness_fn(self, board):
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
        self.fitnesses = []
        self.population = self.problem.get_start_population()

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

    def mutate(self, x):
        if np.random.rand() < 0.1:
            x[np.random.randint(8)] = np.random.randint(8)
        return x

    def train(self, num_iterations=100):
        for epoch in trange(num_iterations):
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
            if self.fitnesses[-1] == 29:
                break
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

    def selector(self, population, fitness_fn, size=1):
        weights = [fitness_fn(board) for board in population]
        # return population[np.argmax(weights)]
        probs = [np.exp(weight) / np.sum(np.exp(weights)) for weight in weights]
        indices = np.random.choice(
            [i for i in range(len(population))], size=size, p=probs
        )
        population = np.array(population)
        return population[indices]


class TSPAlgo(GeneticAlgo):
    def __init__(self, problem):
        super().__init__(problem)

    def mutate(self, x):
        idx1 = np.random.randint(len(x))
        idx2 = np.random.randint(len(x))
        temp = x[idx1]
        x[idx1] = x[idx2]
        x[idx2] = temp
        return x

    def reproduce(self, x, y):
        idx1 = np.random.randint(x.shape[1])
        idx2 = idx1 + 2
        res = np.ones_like(x)
        print(x.shape)
        print(res)
        res *= -1
        res[idx1:idx2] = x[idx1:idx2]
        i = 0
        for e in y:
            if e not in res and i not in range(idx1, idx2):
                res[i] = e
                i = (i + 1 + len(x)) % len(x)

        return res

    def selector(self, population, fitness_fn, size=1):
        weights = [fitness_fn(board) for board in population]
        # return population[np.argmax(weights)]
        probs = [weight / sum(weights) for weight in weights]
        indices = np.random.choice(
            [i for i in range(len(population))], size=size, p=probs
        )
        population = np.array(population)
        return population[indices]


if __name__ == "__main__":

    num_counts = []
    for i in trange(1):
        n_queens = NQueens()
        my_genetic_algo = GeneticAlgo(n_queens)
        my_genetic_algo.train(int(sys.argv[1]))
        my_genetic_algo.plot_fitnesses()
        num_counts.append(len(my_genetic_algo.fitnesses))

    fig, ax = plt.subplots(1, 1)
    a = num_counts
    ax.hist(
        a, bins=[1, 500, 1000, 1500, 2000, 2500, 3000, 3500, 4000, 4500, 5000, 6000]
    )
    ax.set_title("Histogram of result")

    ax.set_xlabel("Generations")
    ax.set_ylabel("Number of times")
    plt.show()
    # fast_algo = FastGeneticAlgo(n_queens)
    # fast_algo.train(int(sys.argv[1]))
    # my_genetic_algo.plot_fitnesses()

    # sns.set_context("paper")
    # plt.plot(my_genetic_algo.fitnesses, label="Slowww")
    # plt.plot(fast_algo.fitnesses, label="FASTTTTT")
    # plt.legend(loc="best")
    # # my_genetic_algo.train()
    # plt.title("Genetic Algorithm on nQueens")
    # plt.show()
