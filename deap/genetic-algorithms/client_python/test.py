"""
Genetic Algorithm Program - December 2017
Written By Jie Jenny Yan with help from:
http://deap.readthedocs.io/en/master/examples/ga_onemax.html
https://github.com/DEAP/deap/blob/master/examples/ga/onemax.py

Using DEAP, we are going to solve a simple parabaloid using a Genetic Algorithm.

A Genetic Algorithm is a method of finding an optimum of a population through
survival of the fittest style evolution.

The code was originally modified for the Black Box Competition functions, so
the evaluation function is still written in such a way that it conforms to the
requirements of using the Black Box Competition functions.

Sections were modified to write information about the program to a file, i.e
which variables have been modified so that they can be compared later on.
"""

from numpy import *
import sys
import platform
import json

from deap import base
from deap import creator
from deap import tools

MAXGEN = 400
MU = 0
SIGMA = 0.5
INDPB = 0.05
PADDING = 4
POPSIZE = 200
TOURNSIZE = 3
DIM = 2
OBJ = 1

# -----------------------------------------------------------------------
# Initialising the tools to be used for convenience and readability

creator.create("Fitness", base.Fitness, weights=(1.0,))
# Points type for the inputs to the function
creator.create("Points", ndarray)
creator.create("Individual", ndarray, fitness=creator.Fitness)

toolbox = base.Toolbox()
# creates the attribute points to be stored in the individual with 2 random points
toolbox.register("attr_points", tools.initRepeat, creator.Points, random.rand, DIM)

# creates an individual -- consider the '1'...
toolbox.register("individual", tools.initRepeat, creator.Individual,
                 toolbox.attr_points, OBJ)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

# ----------------------------------------------------------------------
# function that takes an individual and evaluates the corresponding z value
# for the given x, y points of the individual
def evalparabola(individual):
    individual.fitness.values = zeros(OBJ)
    values = array(individual.fitness.values)
    values[0] = -individual[0][0]**2 - individual[0][1]**2
    individual.fitness.values = values
    f.write("{},{}\n".format(individual, individual.fitness.values))
    return values

# assign the relevant functions to be used by the genetic algorithm program
toolbox.register("evaluate", evalparabola)
toolbox.register("mate", tools.cxOnePoint)
toolbox.register("mutate", tools.mutGaussian, mu=MU, sigma=SIGMA, indpb=INDPB)
toolbox.register("select", tools.selTournament, tournsize=TOURNSIZE)

# main function that runs the genetic algorithm
def main():
    random.seed(64)

    pop = toolbox.population(n=POPSIZE)

    CXPB, MUTPB = 0.5, 0.2

    f.write("----------------------------------------------------------\n")
    f.write("-------------------Start of evolution---------------------\n")
    f.write("----------------------------------------------------------\n")
    f.write("Description:\n")
    f.write(PADDING * " " + "Function: parabaloid\n")
    f.write(PADDING * " " + "Dim size: {}\n".format(DIM))
    f.write(PADDING * " " + "Obj Size: {}\n".format(OBJ))
    f.write("Variables:\n")
    f.write(PADDING * " " + "Mutation: mutGaussian\n")
    f.write(PADDING * " " + "Selection: selTournament\n")
    f.write(PADDING * " " + "Mean: {}\n".format(MU))
    f.write(PADDING * " " + "Std Dev: {}\n".format(SIGMA))
    f.write(PADDING * " " + "Indpb Mutation: {}\n".format(INDPB))
    f.write(PADDING * " " + "Population Size: {}\n".format(POPSIZE))
    f.write(PADDING * " " + "Generations: {}\n".format(MAXGEN))
    f.write(PADDING * " " + "CXPB, MUTPB: {}, {}\n".format(CXPB, MUTPB))
    f.write(PADDING * " " + "Selection tournsize: {}\n".format(TOURNSIZE))

    # Evaluate the entire population
    fitnesses = list(map(toolbox.evaluate, pop))
    for ind, fit in zip(pop, fitnesses):
        ind.fitness.values = fit

    # Extracting all the fitnesses from the individuals
    fits = [ind.fitness.values[0] for ind in pop]

    g = 0
    while g < MAXGEN:
        g = g + 1
        f.write("-- Generation %i --\n" % g)

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

        f.write("  Min %s\n" % min(fits))
        f.write("  Max %s\n" % max(fits))
        f.write("  Avg %s\n" % mean)
        f.write("  Std %s\n" % std)

    best_ind = tools.selBest(pop, 1)[0]
    f.write("Best individual is %s, with value %s\n" % (best_ind, best_ind.fitness.values))
    f.write("----------------------------------------------------------\n")
    f.write("---------------------End of Evolution---------------------\n")
    f.write("----------------------------------------------------------\n")

if __name__ == "__main__":
    f = open("data1.txt", "a")
    main()
    f.close()
