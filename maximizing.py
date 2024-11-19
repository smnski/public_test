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

def main():
    # Przypisanie algorytmu SimpleEvolution z EC-KitY i nadanie mu odpowiednich parametrow
    algo = SimpleEvolution(
        Subpopulation(
            # W pierwszej subpopulacji tworzymy punkty z zakresu [-10, 10]
            creators=GAFloatVectorCreator(length=1, bounds=(-10, 10)),
            population_size=50,
            evaluator=MyEvaluator(),
            # True, poniewaz problem maksymalizacji
            higher_is_better=True,
            # Niewielki elityzm, aby kilka najlepszych jednostek przeszlo dalej, jednoczesnie wciaz eksplorujac nowe rozwiazania
            elitism_rate=0.05,
            # W kazdej iteracji odbywa sie Blend Crossover oraz szansa 10% na dodatkowa mutacje w zakresie 0.2 w rozkladzie Gaussa
            operators_sequence=[
                BlendCrossover(probability=0.7, alpha=0.5),  # Blend crossover
                FloatVectorGaussNPointMutation(probability=0.1, sigma=0.2)
            ],
            # Metoda wyboru tournament - Starcia miedzy pojedynczymi jednostkami i wybor najlepszej
            selection_methods=[
                (TournamentSelection(tournament_size=4), 1)
            ]
        ),
        breeder=SimpleBreeder(),
        max_workers=4,
        max_generation=50,
        statistics=BestAverageWorstStatistics()
    )

    # Ewolucja populacji
    algo.evolve()

    # Uruchomienia algorytmu i zapisanie najlepszego rozwiazania
    best_solution = algo.execute()
    print(f"Best solution found: {best_solution}")

if __name__ == '__main__':
    main()