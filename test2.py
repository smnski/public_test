# Potrzebne narzedzia z EC-KitY
from eckity.algorithms.simple_evolution import SimpleEvolution
from eckity.breeders.simple_breeder import SimpleBreeder
from eckity.creators.ga_creators.float_vector_creator import GAFloatVectorCreator
from eckity.genetic_operators.mutations.vector_random_mutation import FloatVectorGaussNPointMutation
from eckity.genetic_operators.selections.tournament_selection import TournamentSelection
from eckity.statistics.best_average_worst_statistics import BestAverageWorstStatistics
from eckity.subpopulation import Subpopulation
from eckity.evaluators.simple_individual_evaluator import SimpleIndividualEvaluator
from eckity.genetic_operators.genetic_operator import GeneticOperator

# Inne potrzebne biblioteki
import random
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Zakres na jakim prowadzimy maksymalizacje
LOWER_BOUND = -100
UPPER_BOUND = 100
subpops_invis = []

# Funkcja utrzymujaca wartosci w zadanym przedziale
def check_bounds(individual):
    value = individual.cell_value(0)
    bounded_value = min(max(value, LOWER_BOUND), UPPER_BOUND)
    individual.set_cell_value(0, bounded_value)

# Klasa obliczajaca przystosowanie
class MyEvaluator(SimpleIndividualEvaluator):
    def evaluate_individual(self, individual):
        x = individual.cell_value(0)
        return x**2
    
# Klasa przeprowadzajaca Blend crossover na wartosciach rodzicow
class BlendCrossover(GeneticOperator):
    def __init__(self, probability, alpha=0.5):
        super().__init__(probability=probability, arity=2)
        self.alpha = alpha

    def apply(self, individuals):
        parent1, parent2 = individuals
        x1 = parent1.cell_value(0)
        x2 = parent2.cell_value(0)

        offspring1 = parent1.clone()
        offspring2 = parent2.clone()

        gamma = (1. + 2. * self.alpha) * random.random() - self.alpha
        new_x1 = (1. - gamma) * x1 + gamma * x2
        new_x2 = gamma * x1 + (1. - gamma) * x2

        offspring1.set_cell_value(0, new_x1)
        offspring2.set_cell_value(0, new_x2)

        check_bounds(offspring1)
        check_bounds(offspring2)

        return [offspring1, offspring2]

# Wrapper for SimpleEvolution class
class SimpleEvolutionWrapper(SimpleEvolution):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Initialize an empty list to store the best individual of each generation
        self.best_individuals = []

    def generation_iteration(self, gen):
        """
        Performs one iteration of the evolutionary run, for the current generation.

        Parameters
        ----------
        gen:
            Current generation number (for example, generation #100)

        Returns
        -------
        None.
        """
        # Call the original method to keep the default functionality
        super().generation_iteration(gen)

        # Save the best individual of this generation to the list
        print(self.population.sub_populations[0].individuals[0].get_pure_fitness())
        subpops_invis.append(self.population.sub_populations[0].individuals)
        self.best_individuals.append(self.best_of_gen)

    def get_best_individuals(self):
        """
        Return the list of best individuals from each generation.
        """
        return self.best_individuals

# Main function to run the algorithm
def main():
    # Przypisanie algorytmu SimpleEvolutionWrapper z EC-KitY i nadanie mu odpowiednich parametrow
    algo = SimpleEvolutionWrapper(
        Subpopulation(
            creators=GAFloatVectorCreator(length=1, bounds=(-10, 10)),
            population_size=10,
            evaluator=MyEvaluator(),
            higher_is_better=True,
            elitism_rate=0.05,
            operators_sequence=[
                BlendCrossover(probability=0.7, alpha=0.5),
                FloatVectorGaussNPointMutation(probability=0.1, sigma=0.2)
            ],
            selection_methods=[(TournamentSelection(tournament_size=4), 1)]
        ),
        breeder=SimpleBreeder(),
        max_workers=4,
        max_generation=5,
        statistics=BestAverageWorstStatistics()
    )

    # Ewolucja populacji
    algo.evolve()

    # Uruchomienia algorytmu i zapisanie najlepszego rozwiazania
    best_solution = algo.execute()
    print(f"Best solution found: {best_solution}")

    # Print best individuals from each generation
    # best_individuals = algo.get_best_individuals()
    # print("Best individuals from each generation:")
    # for idx, individual in enumerate(best_individuals):
    #     print(f"Generation {idx + 1}: {individual.get_pure_fitness()}")

    def f(x):
        return x**2

    for subpop_invis in subpops_invis:
        for x in subpop_invis:
            print(x.get_pure_fitness())
        print("---------------------------")

    for subpop_invis in subpops_invis:
            x_points = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]  # You can change this list as needed
            y_points = [x.get_pure_fitness() for x in subpop_invis]
            x = np.linspace(-10, 10, 400)
            y = f(x)
            plt.plot(x, label=r'$y = x^2$', color='blue')  # Plot the x^2 function
            plt.scatter(x_points, y_points, color='red', zorder=5)
            for (x_val, y_val) in zip(x_points, y_points):
                plt.text(x_val, y_val, f'({x_val}, {y_val})', fontsize=9, ha='right')
            plt.xlabel('x')
            plt.ylabel('y')
            plt.title('Plot of $y = x^2$ with points')
            plt.grid(True)
            plt.legend()
            plt.show()
if __name__ == '__main__':
    main()