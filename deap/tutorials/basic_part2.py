import random

from deap import base
from deap import creator
from deap import tools

IND_SIZE = 5

# You can create fitnessmin using negative weights
creator.create("FitnessMax", base.Fitness, weights(1.0, 1.0))
creator.create("Individual", list, fitness=creator.FitnessMax)

toolbox = base.Toolbox()
# assigns an alias to a function. Subsequent arguments will become
# arguments of the given function
toolbox.register("attr_float", random.random)
toolbox.register("individual", tools.initRepeat, creator.Individual,
                 toolbox.attr_flaot, n=IND_SIZE)

# to create an individual, we just write:
ind1 = toolbox.individual()             # this var holds a list of flaots

# ----------------------Evaluation ----------------------------
# return a tuple that represents the fitness value
# single objective fitness still requires a tuple because it is considered
# a special case of multi-objective

# ----------------------Mutation-------------------------------
# mutation operator mutates the object on the spot, so need to make sure that
# if you're using a reference to an object you want/need to keep that you
# make a copy/clone of the object. Surely, this is just standard privacy and
# security anyway, so just do it

# ----------------------Crossover------------------------------
