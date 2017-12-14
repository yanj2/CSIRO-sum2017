"""
Using DEAP, we are going to solve the One Max Problem using a Genetic Algorithm.

A Genetic Algorithm is a method of finding an optimum of a population through
survival of the fittest style evolution.

One Max Problem: We want to create a population of individuals
consisting of integer vectors randomly filled with 0 and 1.
We want our population to evolve until one of its members contains only
1 and no 0's anymore

This is written with guidance from:
http://deap.readthedocs.io/en/master/examples/ga_onemax.html
https://github.com/DEAP/deap/blob/master/examples/ga/onemax.py
"""
import random

from deap import base
from deap import creator
from deap import tools

# After they've been created, all our defined classes will be part of the
# creator container
creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Individual", list, fitness=creator.FitnessMax)

toolbox = base.Toolbox()
# Attribute generator
toolbox.register("attr_bool", random.randint, 0, 1)
# Structure initialisers
toolbox.register("individual", tools.initRepeat, creator.Individual,
                 toolbox.attr_bool, 100)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

def evalOneMax(individual):
    # returns an iterable of equal length to the no. of objectives/weights
    return sum(individual),

toolbox.register("evaluate", evalBbcomp)
toolbox.register("mate", tools.cxTwoPoint)
toolbox.register("mutate", tools.mutFlipBit, indpb=0.05)
toolbox.register("select", tools.selTournament, tournsize=3)

def evalBbcomp(individual):
    value = 0
    bbcomp.evaluate(individual, value)
    return value

def main():
    random.seed(64)

    pop = toolbox.population(n=300)

    CXPB, MUTPB = 0.5, 0.2

    print("Start of evolution")

    # Evaluate the entire population
    fitnesses = list(map(toolbox.evaluate, pop))
    for ind, fit in zip(pop, fitnesses):
        ind.fitness.values = fit

    # Extracting all the fitnesses of
    fits = [ind.fitness.values[0] for ind in pop]

    g = 0

    while max(fits) < 100 and g < 1000:
        g = g + 1
        print("-- Generation %i --" % g)

        # Select the next generation individuals
        offspring = toolbox.select(pop, len(pop))
        offspring = list(map(toolbox.clone, offspring))

        # Apply crossover and mutation on the offspring
        for child1, child2 in zip(offspring[::2], offspring[1::2]):
             if random.random() < CXPB:
                 toolbox.mate(child1, child2)
                 del child1.fitness.values
                 del child2.fitness.values

        for mutant in offspring:
            if random.random() < MUTPB:
                toolbox.mutate(mutant)
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
    print("Best individual is %s, %s" % (best_ind, best_ind.fitness.values))

if __name__ == "__main__":
    main()
