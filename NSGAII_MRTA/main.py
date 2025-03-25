import src.utils as utils
import src.evolution as evo


def main():
    number_of_repetitions = 3
    number_of_iterations = 1000
    problem_variation = 1       # Options: src:bi-objective MinMax SD-MTSP
                                #          2:bi-objective SD-MTSP
                                #           with objectives->minimize total
                                #           cost, minimize sum of differences
                                #           between individual tour times and
                                #           avg. tour time.
    number_of_tours = 5
    population_size = 100
    selection_probability = 1
    crossover_type = 'hx'       # Options:  'pmx':partially-mapped,
                                #           'cycx':cyclic,
                                #           'ox':ordered,
                                #           'hx':heirarchical
    mutation_probability = 0.05 # Options: 0<mutation_probability<=src
    tournament_bracket_size = 2 # Note: is number of competing individuals
                                #       in tournament round (2=>Binary)


    instance_file = './data/ch150.txt'

    save_fd = input("Save plot and final generation fronts?(y/n) [default:n]:")

    print("Reading instance from file:", instance_file, "...")
    C, data, T = utils.readInstance(instance_file)
    number_of_cities = C.shape[0]

    print("\n-------------------- PROGRAM START --------------------\n")
    print("*** Multi-Objective Problem: MOmTSP with", number_of_cities,
          "cities and", number_of_tours, "salespersons ***\n")

    print("PARAMETERS: \nCrossover type->", crossover_type,
          "; Mutation Probability->", mutation_probability,
          "; n(iterations)->", number_of_iterations)
    print("Population size->", population_size)

    best_front, fronts = evo.evolve(population_size, C, T, data, number_of_tours,
                                    number_of_iterations, crossover_type,
                                    selection_probability,
                                    mutation_probability, tournament_bracket_size,
                                    number_of_repetitions)
    if save_fd == 'y':
        utils.saveData(fronts)
        utils.plotAndSaveFigures(best_front, population_size, problem_variation,
                                 'y')
    else:
        utils.plotAndSaveFigures(best_front, population_size, problem_variation)
    print("\n--------------------- PROGRAM END ---------------------\n")


if __name__ == "__main__":
    main()