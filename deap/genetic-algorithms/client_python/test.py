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
                 toolbox.attr_points, 1)
toolbox.register("parents", tools.initRepeat, list, toolbox.individual)
toolbox.register("mate", tools.cxOnePoint)


parents = toolbox.parents(4)
print(parents)
toolbox.mate(parents[0][0], parents[1][0])
print(parents)
quit()
person.fitness.values = [2]
print(person, person.fitness.values)
