import sys

# mpl.rcParams["figure.dpi"] = 600
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from tqdm import trange

sns.set()
# np.random.seed(24)


class Problem:
    """Abstract class to define the requirements of any problem"""

    def __init__(self):
        pass

    def get_initial_population(self):
        """Returns the initial population to the algorithm"""
        raise NotImplementedError()

    def fitness_fn(self, individual):
        """Returns the fitness value for an instance of the population"""
        raise NotImplementedError()


class TSP(Problem):
    def __init__(self):
        super().__init__()
        self.distances = {}
        INF = 10

        for i in range(14):
            self.distances[chr(ord("A") + i)] = {}

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

    def get_initial_population(self):
        """Returns the start population, of 20 identical tours
        State representation of every tour is a list with every element as a city of the tour.
        """
        path = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M", "N"]
        list_population = [path for i in range(20)]
        return np.stack(list_population, axis=0)

    def fitness_fn(self, tour):
        """Returns the fitness value of a tour.
        fitness is defined as 1/path_cost

        Args:
            tour (List): An individual tour of the population
        """
        res = self.distances[tour[-1]][tour[0]]
        for i in range(len(tour) - 1):
            res += self.distances[tour[i]][tour[i + 1]]
        return 1 / res


class NQueens(Problem):
    def __init__(self):
        super().__init__()

    def get_initial_population(self):
        """Returns the initial population of 8 identical boards"""
        random_val = np.random.randint(8)
        list_population = [np.ones(8, dtype=int) * random_val for i in range(20)]
        return np.stack(list_population, axis=0)

    def fitness_fn(self, board):
        """Returns the fitness_value of an indiviual board configuration
        fitness is defined as 1 + number_of_conflicting_queens

        Args:
            board (List): An individual instance of the board
        """
        board = np.array(board)
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
    """The Most Basic form of Genetic Algorithm"""

    def __init__(self, problem):
        """
        Args:
            problem (Problem): The problem object to run the algorithm on
        """
        self.problem = problem
        self.population = self.problem.get_initial_population()
        self.fitnesses = [
            np.max(
                np.array(
                    [self.problem.fitness_fn(member) for member in self.population]
                )
            )
        ]
        self.best_individual = ([], -1)

    def selector(self, population):
        """This method is responsible for selecting an individual from the population for reproducing

        Args:
            population (List): A 2D List containing individuals in different rows
            fitness_fn
        """
        weights = np.array([self.problem.fitness_fn(board) for board in population])
        probs = [weight / sum(weights) for weight in weights]
        # probs = [np.exp(weight)/np.sum(np.exp(weights)) for weight in weights]
        # probs = [weight*weight for weight in weights]
        # probs = [prob/sum(probs) for prob in probs]
        indices = np.random.choice([i for i in range(len(population))], p=probs)
        population = np.array(population)
        return population[indices]

    def reproduce(self, x, y):
        """Function responsible for creating a child from when given two parents
        Returns the child

        Args:
            x : Parent1
            y : Parent2
        """
        slice_idx = np.random.randint(8)
        new_guy = np.array(x[: int(slice_idx)])
        return np.append(new_guy, np.array(y[int(slice_idx) :]))

    def mutate(self, x, epoch=None):
        """Responsible for mutating an individual
        Args:
            x : Individual to mutate
        """
        if np.random.rand() < 0.1:
            x[np.random.randint(8)] = np.random.randint(8)
        return x

    def train(self, num_generations=100):
        """The core of the algoritm, this function is responsible for making new generations of the population
        Args:
            num_generations (Int): Number of generations to produce
        """
        with trange(num_generations) as t:
            for epoch in t:
                new_population = []
                for i in range(len(self.population)):
                    x = self.selector(self.population)
                    y = self.selector(self.population)
                    child = self.reproduce(x, y)
                    child = self.mutate(child, epoch=epoch)
                    new_population.append(child)

                self.population = np.array(new_population)
                best_individual = np.argmax(
                    np.array(
                        [self.problem.fitness_fn(member) for member in new_population]
                    )
                )

                self.fitnesses.append(
                    np.max(
                        np.array(
                            [
                                self.problem.fitness_fn(member)
                                for member in new_population
                            ]
                        )
                    )
                )
                if self.fitnesses[-1] > self.best_individual[1]:
                    self.best_individual = (best_individual, self.fitnesses[-1])

                t.set_postfix(
                    ordered_dict={f"{self.__class__.__name__}": self.fitnesses[-1]}
                )

    def plot_fitnesses(self):
        """Utility function to visualize the results, plots the fitness v/s generations plot"""
        sns.set_palette("colorblind")
        sns.set_context("paper")
        plt.plot(self.fitnesses)
        plt.title(f"{self.__class__.__name__} on {self.problem.__class__.__name__}")
        plt.show()


class FastGeneticAlgo(GeneticAlgo):
    def __init__(self, problem):
        super().__init__(problem)

    def reproduce(self, x, y):
        candidates = []
        for slice_idx in range(8):
            candidate1 = np.concatenate([x[:slice_idx], y[slice_idx:]])
            candidate2 = np.concatenate([x[: 8 - slice_idx], y[8 - slice_idx :]])
            candidates.append(candidate1)
            candidates.append(candidate2)
        return self.greedy_selector(np.array(candidates), self.problem.fitness_fn)

    def greedy_selector(self, population, fitness_fn):
        weights = [fitness_fn(board) for board in population]
        idx = np.argmax(weights)
        return population[idx]

    def selector(self, population):
        weights = [self.problem.fitness_fn(board) for board in population]
        probs = [np.exp(weight) / np.sum(np.exp(weights)) for weight in weights]
        indices = np.random.choice([i for i in range(len(population))], p=probs)
        return population[indices].flatten()

    def mutate(self, x, epoch=None):
        force = (
            epoch in range(200, 400) and self.fitnesses[-1] == self.fitnesses[-10]
            if epoch
            else False
        )
        if np.random.rand() < 0.1 or force:
            x[np.random.randint(8)] = np.random.randint(8)
        return x


class GenCheck(FastGeneticAlgo):
    def __init__(self, problem):
        super().__init__(problem)

    def selector(self, population):
        """This method is responsible for selecting an individual from the population for reproducing

        Args:
            population (List): A 2D List containing individuals in different rows
            fitness_fn
        """
        weights = np.array([self.problem.fitness_fn(board) for board in population])
        probs = [weight / sum(weights) for weight in weights]
        # probs = [np.exp(weight)/np.sum(np.exp(weights)) for weight in weights]
        # probs = [weight*weight for weight in weights]
        # probs = [prob/sum(probs) for prob in probs]
        indices = np.random.choice([i for i in range(len(population))], p=probs)
        population = np.array(population)
        return population[indices].flatten()


class TSPAlgo(GeneticAlgo):
    def __init__(self, problem):
        super().__init__(problem)

    def mutate(self, x, epoch=None):
        if np.random.rand() < 0.2:
            idx1 = np.random.randint(x.shape[0])
            idx2 = np.random.randint(x.shape[0])
            x[idx1], x[idx2] = x[idx2], x[idx1]
        return x

    def reproduce(self, x, y):
        start_idx = np.random.randint(x.shape[0])
        end_idx = np.random.randint(x.shape[0])
        if start_idx > end_idx:
            start_idx, end_idx = end_idx, start_idx

        res = np.array([chr(ord("X")) for i in range(x.shape[-1])])

        res[start_idx:end_idx] = x[start_idx:end_idx]

        for i in range(len(y)):
            if y[i] not in res:
                for ii in range(len(res)):
                    if res[ii] == "X":
                        res[ii] = y[i]
                        break
        return res


class CycleTSP(TSPAlgo):
    def __init__(self, problem):
        super().__init__(problem)

    def mutate(self, x, epoch=None):
        force = (
            epoch > 400 and self.fitnesses[-1] == self.fitnesses[-10]
            if epoch
            else False
        )
        if np.random.rand() < 0.2 or force:
            idx1 = np.random.randint(x.shape[0])
            idx2 = np.random.randint(x.shape[0])
            x[idx1], x[idx2] = x[idx2], x[idx1]
        return x

    def reproduce(self, p1, p2):
        n = p1.shape[0]
        p1_real = p1.copy().tolist()
        p2_real = p2.copy().tolist()
        p1 = p1.copy().tolist()
        p2 = p2.copy().tolist()
        child1 = ["X" for i in range(n)]
        child2 = ["X" for i in range(n)]
        cycle = set()
        count = 0
        i = 0
        cycle.add(p1[0])
        p1[0] = -1
        cycles = []
        # print(cycle)
        while count < n:
            if p2[i] in cycle:
                # print(f"i:{i}, p2[i]: {p2[i]}, cycle: {cycle}")
                cycles.append((cycle.copy()))
                p2[i] = -1
                cycle.clear()
                for j in range(n):
                    if p1[j] != -1:
                        i = j
                        break
                cycle.add(p1[i])
                count += 1
                p1[i] = -1
                continue
            i_new = p1.index(p2[i])
            p2[i] = -1
            i = i_new
            cycle.add(p1[i])
            count += 1
            p1[i] = -1
        # print(cycles)
        p1 = p1_real
        p2 = p2_real
        for i in range(n):
            for idx, cycle in enumerate(cycles):
                if p1[i] in cycle:
                    if idx % 2 == 0:
                        child1[i] = p1[i]
                    else:
                        child2[i] = p1[i]
        for i in range(n):
            for idx, cycle in enumerate(cycles):
                if p2[i] in cycle:
                    if idx % 2 == 0:
                        child2[i] = p2[i]
                    else:
                        child1[i] = p2[i]
        child1 = np.array(child1)
        child2 = np.array(child2)
        return self.greedy_selector(np.array([child1, child2]), self.problem.fitness_fn)

    def greedy_selector(self, population, fitness_fn):
        weights = [fitness_fn(board) for board in population]
        idx = np.argmax(weights)
        return population[idx]


class FastTSP(TSPAlgo):
    def __init__(self, problem):
        super().__init__(problem)

    def mutate(self, x, epoch=None):
        res = (super().mutate(x, epoch=epoch)).tolist()
        rotated_res = res[:]
        rotate_idx = np.random.randint(len(res))
        rotated_res[rotate_idx:] = res[:-rotate_idx]
        rotated_res[:rotate_idx] = res[-rotate_idx:]
        assert np.allclose(
            self.problem.fitness_fn(res), self.problem.fitness_fn(rotated_res)
        )
        return np.array(rotated_res)

    def reproduce(self, x, y):
        candidates = []
        for start_idx in range(x.shape[0]):
            end_idx = np.random.randint(x.shape[0])
            if start_idx > end_idx:
                start_idx, end_idx = end_idx, start_idx

            res = np.array([chr(ord("X")) for i in range(x.shape[-1])])

            res[start_idx:end_idx] = x[start_idx:end_idx]

            lookup_set = set(x[start_idx:end_idx])

            ii = end_idx

            for e in y:
                if e in lookup_set:
                    continue
                res[ii] = e
                ii = (ii + 1 + res.shape[0]) % res.shape[0]

            candidates.append(res)

        return self.greedy_selector(candidates, self.problem.fitness_fn)

    def greedy_selector(self, population, fitness_fn):
        weights = [fitness_fn(board) for board in population]
        idx = np.argmax(weights)
        return population[idx]


def plot_comparison(
    optimized_algo, original_algo, title, saveLocation="./", numRuns=None
):
    print(len(original_algo), len(original_algo[0]))
    sns.set_palette("colorblind")
    sns.set_context("paper")
    plt.suptitle(title, fontsize="x-large")
    if numRuns:
        plt.title(f"Averaged over {i+1} runs")
    plt.xlabel("#Generation")
    plt.ylabel("Max Fitness of the Population")
    plt.yticks(np.arange(0, 31, step=2))
    plt.plot(np.mean(optimized_algo, axis=0), label="Optimized Algorithm")
    plt.plot(np.mean(original_algo, axis=0), label="Original Algorithm")
    plt.legend(loc="best")
    figName = f"{saveLocation.split('/')[-1]}{numRuns+1 if numRuns else 'image'}.png"
    plt.savefig(f"{saveLocation}/{figName}")
    plt.clf()


if __name__ == "__main__":

    # tsp = TSP()
    # tsp_gen = TSPAlgo(tsp)
    # tsp_gen.train(1000)
    # tsp_gen.plot_fitnesses()
    # sys.exit(0)

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
            t.set_postfix(
                maxFast=max(fast_genetic_algo.fitnesses),
                maxSlow=max(my_genetic_algo.fitnesses),
                gFast=len(fast_genetic_algo.fitnesses),
                gSlow=len(my_genetic_algo.fitnesses),
            )
            if (i + 1) % 10 == 0:
                plot_comparison(
                    avg_fitnesses_fast,
                    avg_fitnesses_slow,
                    title="Genetic Algorithm on NQueens",
                    saveLocation="./Plots/GenNQueens2",
                    numRuns=i,
                )
                # sys.exit(0)
