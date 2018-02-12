import numpy as np

from deap import base
from deap import benchmarks
from deap import creator
from deap import tools

from scoop import futures
DIM = 2
# Creates a fitness object that minimises its fitness value
creator.create("Fitness", base.Fitness, weights=(1.0,))

# Creates a particle with initial declaration of its contained attributes
creator.create("Particle", np.ndarray, fitness=creator.Fitness, velocity=np.ndarray(DIM), best_known=None)

def sphere(individual):
    try:
        individual = (individual - 3.)**2
        return -1.0 * (individual[0] + individual[1]),
    except:
        print(individual)

# generates and returns a particle based on the dim (size) of the problem
def generate(bound_l, bound_u):
    particle = creator.Particle(np.random.uniform(bound_l,bound_u) for _ in range(DIM))
    bound = bound_u - bound_l
    particle.velocity = np.array([np.random.uniform(-abs(bound), abs(bound)) for _ in range(DIM)])
    particle.best_known = creator.Particle(particle)
    return particle

toolbox = base.Toolbox()
toolbox.register("evaluate", sphere)
toolbox.register("particle", generate, bound_l=-5, bound_u=5)
toolbox.register("population", tools.initRepeat, list, toolbox.particle)
toolbox.register("map", futures.map)

def fun(n):
    return n * 2,

def test(num):
    num.fitness.values = toolbox.evaluate(num)
    return num.fitness.values

if __name__ == "__main__":
    pop = toolbox.population(n=5)
    print(list(toolbox.map(test, pop)))
