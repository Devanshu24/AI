import sys

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

    def fitness_fn(self, state):
        """Returns the fitness_value of an indiviual board configuration
        fitness is defined as 1 + number_of_conflicting_queens

        Args:
            state (List): An individual instance of the board
        """
        state = np.array(state)
        n = state.shape[-1]
        row_freq = [0] * n
        diag_freq = [0] * (2 * n)
        diag_freq_2 = [0] * (2 * n)

        for i in range(n):
            row_freq[state[i]] += 1
            diag_freq[state[i] + i] += 1
            diag_freq_2[n - state[i] + i] += 1

        conflicts = 0

        for i in range(2 * n):
            if i < n:
                conflicts += (row_freq[i] * (row_freq[i] - 1)) / 2
            conflicts += (diag_freq[i] * (diag_freq[i] - 1)) / 2
            conflicts += (diag_freq_2[i] * (diag_freq_2[i] - 1)) / 2
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
            epoch (int): the current epoch of training
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

    def greedy_selector(self, population, fitness_fn):
        """An utility function that gives the best candidate from a population based on fitness values"""
        weights = [fitness_fn(board) for board in population]
        idx = np.argmax(weights)
        return population[idx]

    def plot_fitnesses(self):
        """Utility function to visualize the results, plots the fitness v/s generations plot"""
        sns.set_palette("colorblind")
        sns.set_context("paper")
        plt.plot(self.fitnesses)
        plt.title(f"{self.__class__.__name__} on {self.problem.__class__.__name__}")
        plt.show()


class FastGeneticAlgo(GeneticAlgo):
    """An Optimized version of the Genetic Algoritm on NQueens"""

    def __init__(self, problem):
        super().__init__(problem)

    def reproduce(self, x, y):
        """Modified reproduce, instead of generating children with random slicing it generates children based on all possible splices
        And returns the fittest amongst them

        Args:
            x (List): Parent 1
            y (List): Parent 2
        """
        candidates = []
        for slice_idx in range(8):
            candidate1 = np.concatenate([x[:slice_idx], y[slice_idx:]])
            candidate2 = np.concatenate([x[: 8 - slice_idx], y[8 - slice_idx :]])
            candidates.append(candidate1)
            candidates.append(candidate2)
        return self.greedy_selector(np.array(candidates), self.problem.fitness_fn)

    def greedy_selector(self, population, fitness_fn):
        """An utility function that gives the best candidate from a population based on fitness values"""
        weights = [fitness_fn(board) for board in population]
        idx = np.argmax(weights)
        return population[idx]

    def selector(self, population):
        """Modified selector, samples based on the softmax fitnesses instead of simply summing over them

        Args:
            population (List): A 2D list of individuals
        """
        weights = [self.problem.fitness_fn(board) for board in population]
        probs = [np.exp(weight) / np.sum(np.exp(weights)) for weight in weights]
        indices = np.random.choice([i for i in range(len(population))], p=probs)
        return population[indices].flatten()

    def mutate(self, x, epoch=None):
        """
        Modified Mutate, with a force option to force mutation on the individual overriding the probability

        Args:
            x (List): Individual to mutate
            epoch (int): epoch
        """
        force = (
            epoch in range(200, 400) and self.fitnesses[-1] == self.fitnesses[-10]
            if epoch
            else False
        )
        if np.random.rand() < 0.1 or force:
            x[np.random.randint(8)] = np.random.randint(8)
        return x


class TSPAlgo(GeneticAlgo):
    """Basic Genetic Algorithm on TSP"""

    def __init__(self, problem):
        super().__init__(problem)

    def mutate(self, x, epoch=None):
        """Mutates an route by swapping two cities in the route

        Args:
            x (List): Route to mutate
            epoch (int): the current epoch of training
        """
        if np.random.rand() < 0.2:
            idx1 = np.random.randint(x.shape[0])
            idx2 = np.random.randint(x.shape[0])
            x[idx1], x[idx2] = x[idx2], x[idx1]
        return x

    def reproduce(self, x, y):
        """Responsible for doing the crossover of two routes to create a new route
        Adds a slice of the route from x to the child in the same place and fills the remaining locations from the second parent y, if they are not already in the child
        Returns the new route

        Args:
            x (List): Parent 1
            y (List): Parent 2
        """
        start_idx = np.random.randint(x.shape[0])
        end_idx = np.random.randint(x.shape[0])
        if start_idx > end_idx:
            start_idx, end_idx = end_idx, start_idx

        res = np.array([chr(ord("X")) for i in range(x.shape[-1])])

        for i in range(len(y)):
            if y[i] not in res:
                for ii in range(len(res)):
                    if res[ii] == "X":
                        res[ii] = y[i]
                        break
        return res

    def greedy_selector(self, population, fitness_fn):
        """An utility function that gives the best candidate from a population based on fitness values"""
        weights = [fitness_fn(board) for board in population]
        idx = np.argmax(weights)
        return population[idx]


class PMXAlgo(TSPAlgo):
    """Genetic Algorithm with PMX Crossover"""

    def __init__(self, problem):
        super().__init__(problem)

    def reproduce(self, x, y):
        """Uses PMX (Partially Mapped Crossover) to reproduce

        Args:
            x (List): Parent 1
            y (List): Parent 2
        """
        x = x.tolist()
        y = y.tolist()

        # Uncomment to use with TSP

        x = [ord(e) - ord("A") for e in x]
        y = [ord(e) - ord("A") for e in y]

        n = len(x)
        p1 = np.zeros(n, dtype=np.int64)
        p2 = np.zeros(n, dtype=np.int64)

        for i in range(n):
            p1[x[i]] = i
            p2[y[i]] = i

        # Choose crossover points
        start_idx = np.random.randint(n)
        end_idx = np.random.randint(n)
        if end_idx < start_idx:
            start_idx, end_idx = end_idx, start_idx

        for i in range(start_idx, end_idx):

            temp1 = x[i]
            temp2 = y[i]

            x[i], x[(p1[(temp2)])] = temp2, temp1
            y[i], y[(p2[(temp1)])] = temp1, temp2

            p1[temp1], p1[temp2] = p1[temp2], p1[temp1]
            p2[temp1], p2[temp2] = p2[temp2], p2[temp1]

        # Uncomment to use with TSP

        x = [chr(e + ord("A")) for e in x]
        y = [chr(e + ord("A")) for e in y]
        x = np.array(x)
        y = np.array(y)

        return self.greedy_selector(np.array([x, y]), self.problem.fitness_fn)


class UPMXAlgo(TSPAlgo):
    def __init__(self, problem):
        super().__init__(problem)

    def reproduce(self, x, y, prob=0.4):
        """Reproduce using UPMX (Uniform Partially Mapped) Crossover"""
        x = x.tolist()
        y = y.tolist()
        # Uncommnent to run on TSP
        x = [ord(e) - ord("A") for e in x]
        y = [ord(e) - ord("A") for e in y]
        n = len(x)
        p1 = np.zeros(n, dtype=np.int64)
        p2 = np.zeros(n, dtype=np.int64)

        for i in range(n):
            p1[x[i]] = i
            p2[y[i]] = i

        for i in range(n):
            if np.random.random() < prob:
                temp1 = x[i]
                temp2 = y[i]

                x[i], x[p1[temp2]] = temp2, temp1
                y[i], y[p2[temp1]] = temp1, temp2

                p1[temp1], p1[temp2] = p1[temp2], p1[temp1]
                p2[temp1], p2[temp2] = p2[temp2], p2[temp1]

        # Uncommnent to run on TSP
        x = [chr(e + ord("A")) for e in x]
        y = [chr(e + ord("A")) for e in y]

        x = np.array(x)
        y = np.array(y)

        return self.greedy_selector(np.array([x, y]), self.problem.fitness_fn)


class OXAlgo(TSPAlgo):
    def __init__(self, problem):
        super().__init__(problem)

    def reproduce(self, x, y):
        """Reproduce using the OX (Ordered) Crossover"""
        x = x.tolist()
        y = y.tolist()

        # Comment to run on NQueens
        x = [ord(e) - ord("A") for e in x]
        y = [ord(e) - ord("A") for e in y]
        n = len(x)

        start_idx = np.random.randint(n)
        end_idx = np.random.randint(n)

        if start_idx > end_idx:
            start_idx, end_idx = end_idx, start_idx

        empt1, empt2 = [True] * n, [True] * n

        for i in range(n):
            if i < start_idx or i > end_idx:
                empt1[y[i]] = False
                empt2[x[i]] = False

        temp1, temp2 = x.copy(), y.copy()

        id1, id2 = end_idx + 1, end_idx + 1

        for i in range(n):
            if not empt1[temp1[(i + end_idx + 1) % n]]:
                x[id1 % n] = temp1[(i + end_idx + 1) % n]
                id1 += 1

            if not empt2[temp2[(i + end_idx + 1) % n]]:
                y[id2 % n] = temp2[(i + end_idx + 1) % n]
                id2 += 1

        for i in range(start_idx, end_idx + 1):
            x[i], y[i] = y[i], x[i]

        # Comment to run on NQueens
        x = [chr(e + ord("A")) for e in x]
        y = [chr(e + ord("A")) for e in y]
        x = np.array(x)
        y = np.array(y)

        return self.greedy_selector(np.array([x, y]), self.problem.fitness_fn)


class FastTSP(TSPAlgo):
    def __init__(self, problem):
        super().__init__(problem)

    def mutate(self, x, epoch=None):
        """Modified mutate function, other than randomly swapping to elements it also rotates the mutated route by some random number of times
        Since the fitness function is indifferent to cyclic permutations, it does not affect the fitness funciton but adds diversity to the population.

        Args:
            x (List): Route to mutate
            epoch (int): The current epoch of the training loop

        """
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
        """Modified reproduce function, instead of randomly chosing both the start and end index of the slice, it iterates over all possible start indices and randomly selecting the end index.
        It swaps them if the start_idx > end_idx

        Args:
            x (List): Parent 1
            y (List): Parent 2
        """
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
        """An utility function that gives the best candidate from a population based on fitness values"""
        weights = [fitness_fn(board) for board in population]
        idx = np.argmax(weights)
        return population[idx]


def plot_comparison(algos, title, saveLocation="./", numRuns=None):
    # print(len(original_algo), len(original_algo[0]))
    print(f"Plotting {i+1}")
    sns.set_palette("colorblind")
    sns.set_context("paper")
    plt.suptitle(title, fontsize="x-large")
    if numRuns:
        plt.title(f"Averaged over {i+1} runs")
    plt.xlabel("#Generation")
    plt.ylabel("Max Fitness of the Population")
    plt.yticks(np.arange(0, 31, step=2))
    for algo, name in algos:
        plt.plot(np.mean(algo, axis=0), label=name)
    plt.legend(loc="best")
    figName = f"{saveLocation.split('/')[-1]}{numRuns+1 if numRuns else 'image'}.png"
    plt.savefig(f"{saveLocation}/{figName}")
    plt.clf()


if __name__ == "__main__":

    print(
        "For running TSP please please enter 'T'(without quotes) for NQueens type 'Q' (without quotes)\nPlease open the code file to run more sophisticated algorithms."
    )
    ss = input()
    print("Enter the number of generations to run the training")
    num_steps = int(input())
    if ss == "T":
        tsp = TSP()
        tsp_gen = TSPAlgo(tsp)
        tsp_gen.train(num_steps)
        tsp_gen.plot_fitnesses()
    elif ss == "Q":
        n_queens = NQueens()
        n_queens_gen = GeneticAlgo(n_queens)
        n_queens_gen.train(num_steps)
        n_queens_gen.plot_fitnesses()
    else:
        print("Please enter a valid choice")

    print("Please open the code file to run more sophisticated algorithms")
    sys.exit(0)

    a1f = []
    a2f = []
    a3f = []
    a4f = []
    a5f = []
    with trange(100) as t:
        for i in t:
            tsp = TSP()
            a1 = TSPAlgo(tsp)
            a2 = PMXAlgo(tsp)
            a3 = UPMXAlgo(tsp)
            a4 = OXAlgo(tsp)
            a5 = FastTSP(tsp)
            a1.train(int(sys.argv[1]))
            a2.train(int(sys.argv[1]))
            a3.train(int(sys.argv[1]))
            a4.train(int(sys.argv[1]))
            a5.train(int(sys.argv[1]))

            a1f.append(a1.fitnesses)
            a2f.append(a2.fitnesses)
            a3f.append(a3.fitnesses)
            a4f.append(a4.fitnesses)
            a5f.append(a5.fitnesses)
            t.set_postfix(
                a1=max(a1.fitnesses),
                a2=max(a2.fitnesses),
                a3=max(a3.fitnesses),
                a4=max(a4.fitnesses),
                a5=max(a5.fitnesses),
            )
            if (i + 1) % 2 == 0:
                plot_comparison(
                    [
                        (a1f, "Original Algorithm"),
                        (a2f, "PMX"),
                        (a3f, "UPMX"),
                        (a4f, "OX"),
                        (a5f, "FastTSP"),
                    ],
                    title="Genetic Algorithm on TSP",
                    numRuns=i,
                )
                # sys.exit(0)
