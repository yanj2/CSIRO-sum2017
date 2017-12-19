from numpy import *
import sys
import platform
import json

from deap import base
from deap import creator
from deap import tools
# -----------------------------------------------------------------------
creator.create("Fitness", base.Fitness, weights=(1.0,))
# Points type for the inputs to the function
creator.create("Points", ndarray)
creator.create("Individual", ndarray, fitness=creator.Fitness)

toolbox = base.Toolbox()
# creates the attribute points to be stored in the individual with `dim` random points
toolbox.register("attr_points", tools.initRepeat, creator.Points, random.rand, 2)

# creates an individual -- consider the '1'...
toolbox.register("individual", tools.initRepeat, creator.Individual,
                 toolbox.attr_points, 1)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

# For the sake of testing while waiting for login details
def evalparabola(individual):
    individual.fitness.values = zeros(1)
    values = array(individual.fitness.values)
    values[0] = -individual[0][0]**2 - individual[0][1]**2
    individual.fitness.values = values
    print("{},{}".format(individual, individual.fitness.values))
    return values


toolbox.register("evaluate", evalparabola)
toolbox.register("mate", tools.cxOnePoint)
#toolbox.register("mutate", tools.mutFlipBit, indpb=0.05)
# use a different mutation method
toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=1.5, indpb=0.1)
toolbox.register("select", tools.selTournament, tournsize=3)

def main():
    random.seed(64)

    pop = toolbox.population(n=200)

    CXPB, MUTPB = 0.5, 0.2

    print("Start of evolution")

    # Evaluate the entire population
    fitnesses = list(map(toolbox.evaluate, pop))
    for ind, fit in zip(pop, fitnesses):
        ind.fitness.values = fit

    # Extracting all the fitnesses from the individuals
    fits = [ind.fitness.values[0] for ind in pop]

    g = 0
    while g < 1000:
        g = g + 1
        print("-- Generation %i --" % g)

        # Select the next generation individuals
        offspring = toolbox.select(pop, len(pop))
        offspring = list(map(toolbox.clone, offspring))

        # Apply crossover and mutation on the offspring
        for child1, child2 in zip(offspring[::2], offspring[1::2]):
             if random.random() < CXPB:
                 toolbox.mate(child1[0], child2[0])
                 del child1.fitness.values
                 del child2.fitness.values

        for mutant in offspring:
            if random.random() < MUTPB:
                toolbox.mutate(mutant[0])
                del mutant.fitness.values

        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        fitnesses = map(toolbox.evaluate, invalid_ind)
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit

        pop[:] = offspring

        fits = [ind.fitness.values[0] for ind in pop]

        length = len(pop)
        mean = sum(fits) / length
        sum2 = sum(x*x for x in fits)
        std = abs(sum2 / length - mean**2)**0.5

        print("  Min %s" % min(fits))
        print("  Max %s" % max(fits))
        print("  Avg %s" % mean)
        print("  Std %s" % std)

    best_ind = tools.selBest(pop, 1)[0]
    print("Best individual is %s, with value %s" % (best_ind, best_ind.fitness.values))

if __name__ == "__main__":
	main()

"""
parents = toolbox.parents(4)
print(parents)
toolbox.mate(parents[0][0], parents[1][0])
print(parents)
quit()
person.fitness.values = [2]
print(person, person.fitness.values)
"""
