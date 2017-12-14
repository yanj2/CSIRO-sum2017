from deap import creator
from deap import base
from deap import tools
from numpy import *

creator.create("Fitness", base.Fitness, weights=(1.0,))
creator.create("Points", ndarray)
creator.create("Individual", ndarray, fitness=creator.Fitness)

toolbox = base.Toolbox()
toolbox.register("attr_points", tools.initRepeat, creator.Points, random.rand, 2)
toolbox.register("individual", tools.initRepeat, creator.Individual,
                 toolbox.attr_points)

person = toolbox.individual(1)

person.fitness.values = [2]
print(person[0], person.fitness.values)
