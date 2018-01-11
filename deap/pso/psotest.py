"""
Particle Swarm Optimization Parabaloid - Written By Jie Jenny Yan, December 2017
https://www.researchgate.net/post/How_can_one_decide_inertia_weight_w_during_the_implementation_of_PSO
"""

import operator
import random

import numpy

from deap import base
from deap import benchmarks
from deap import creator
from deap import tools

# Creates a fitness object that maximises its fitness value
creator.create("FitnessMax", base.Fitness, weights=(1.0,))
# Creates a particle with initial declaration of its contained attributes
creator.create("Particle", list, fitness=creator.FitnessMax, speed=list,
               smin=None, smax=None, best=None)

def evalparabola(individual):
    return -individual[0]**2 - individual[1]**2,

# Function that generates random positions and speeds for a particle
def generate(size, pmin, pmax, smin, smax):
    particle = creator.Particle(random.uniform(pmin, pmax) for _ in range(size))
    particle.speed = [random.uniform(smin, smax) for _ in range(size)]
    particle.smin = smin
    particle.smax = smax
    return particle

# Function that first computes the speed, then limits the speed values between
# smin and smax, and computes the new particle position
def updateParticle(particle, best, phi1, phi2):
    u1 = (random.uniform(0, phi1) for _ in range(len(particle)))
    u2 = (random.uniform(0, phi2) for _ in range(len(particle)))
    v_u1 = map(operator.mul, u1, map(operator.sub, particle.best, particle))
    v_u2 = map(operator.mul, u2, map(operator.sub, best, particle))
    particle.speed = list(map(operator.add, particle.speed,
                     map(operator.add, v_u1, v_u2)))
    for i, speed in enumerate(particle.speed):
        if speed < particle.smin:
            particle.speed[i] = particle.smin
        elif speed > particle.smax:
            particle.speed[i] = particle.smax
    particle[:] = list(map(operator.add, particle, particle.speed))

toolbox = base.Toolbox()
toolbox.register("particle", generate, size=2, pmin=-8, pmax=8, smin=-0.001, smax=0.9)
toolbox.register("population", tools.initRepeat, list, toolbox.particle)
toolbox.register("update", updateParticle, phi1=2.0, phi2=2.0)
toolbox.register("evaluate", evalparabola)
#toolbox.register("evaluate", benchmarks.h1)

def main():
    pop = toolbox.population(n=800)
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", numpy.mean)
    stats.register("std", numpy.std)
    stats.register("min", numpy.min)
    stats.register("max", numpy.max)

    logbook = tools.Logbook()
    logbook.header = ["gen", "evals"] + stats.fields

    GEN = 1000
    best = None

    for g in range(GEN):
        for part in pop:
            part.fitness.values = toolbox.evaluate(part)
            if not part.best or part.best.fitness < part.fitness:
                part.best = creator.Particle(part)
                part.best.fitness.values = part.fitness.values
            if not best or best.fitness < part.fitness:
                best = creator.Particle(part)
                best.fitness.values = part.fitness.values
        for part in pop:
            toolbox.update(part, best)

        # Gather all the fitnesses in one list and print the stats
        logbook.record(gen=g, evals=len(pop), **stats.compile(pop))
        print(logbook.stream)

    return pop, logbook, best

if __name__ == "__main__":
    main()
