from time import process_time
from eckity.algorithms.simple_evolution import SimpleEvolution
from eckity.breeders.simple_breeder import SimpleBreeder
from eckity.creators.ga_creators.float_vector_creator import GAFloatVectorCreator
from eckity.genetic_operators.mutations.vector_random_mutation import FloatVectorGaussNPointMutation
from eckity.genetic_operators.selections.tournament_selection import TournamentSelection
from eckity.statistics.best_average_worst_statistics import BestAverageWorstStatistics
from eckity.subpopulation import Subpopulation
from eckity.evaluators.simple_individual_evaluator import SimpleIndividualEvaluator
from eckity.genetic_operators.genetic_operator import GeneticOperator

# Define the bounds
LOWER_BOUND = -10
UPPER_BOUND = 10

# Define the objective function to minimize (f(x) = x^2)
class MyEvaluator(SimpleIndividualEvaluator):
    def evaluate_individual(self, individual):
        x = individual.cell_value(0)
        return x**2

class BlendCrossover(GeneticOperator):
    def __init__(self, probability, alpha=0.5):
        super().__init__(probability=probability, arity=2)
        self.alpha = alpha

    def apply(self, individuals):
        parent1, parent2 = individuals
        x1 = parent1.cell_value(0)
        x2 = parent2.cell_value(0)

        # Blend crossover calculation
        new_x1 = (1 - self.alpha) * x1 + self.alpha * x2
        new_x2 = self.alpha * x1 + (1 - self.alpha) * x2

        # Create offspring clones and set their new values
        offspring1 = parent1.clone()
        offspring2 = parent2.clone()
        offspring1.set_cell_value(0, new_x1)
        offspring2.set_cell_value(0, new_x2)

        # Ensure bounds are respected using check_bounds function
        check_bounds(offspring1)
        check_bounds(offspring2)

        return [offspring1, offspring2]

def check_bounds(individual):
    value = individual.cell_value(0)
    bounded_value = min(max(value, LOWER_BOUND), UPPER_BOUND)
    individual.set_cell_value(0, bounded_value)

def main():
    start_time = process_time()

    # Initialize the evolutionary algorithm
    algo = SimpleEvolution(
        Subpopulation(
            creators=GAFloatVectorCreator(length=1, bounds=(LOWER_BOUND, UPPER_BOUND)),  # Single-point solution (length=1)
            population_size=50,
            evaluator=MyEvaluator(),
            higher_is_better=True,  # Minimization problem
            elitism_rate=0.1,
            operators_sequence=[
                BlendCrossover(probability=0.7, alpha=0.5),  # Blend crossover
                FloatVectorGaussNPointMutation(probability=0.1, sigma=0.2)  # Mutation with bounds
            ],
            selection_methods=[
                (TournamentSelection(tournament_size=4), 1)  # Tournament selection
            ]
        ),
        breeder=SimpleBreeder(),
        max_workers=4,
        max_generation=500,
        statistics=BestAverageWorstStatistics()
    )

    # Evolve the population
    algo.evolve()

    # Execute and print the best solution
    best_solution = algo.execute()
    print(f"Best solution found: {best_solution}")
    print(f"Total time: {process_time() - start_time}")

if __name__ == '__main__':
    main()