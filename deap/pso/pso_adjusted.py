"""
Particle Swarm Optimisation with improvements - Jie Jenny Yan, January 2018
- Framework: DEAP
- Fitness function: Parabaloid
- Particle Attributes: velocity, best_known, global_best, curr_pos #CHECK
- Constants: upper/lower bounds (b_u, b_l: generate function, random.uniform),
             inertia weighting (w: swarm evolution -> velocity equation),
             accel coefficients (phi_p, phi_g: swarm evolution -> velocity equation),
             diversify search (r_p, r_g: swarm evolution, velocity equation),

PSO Algorithm:

1) swarm initialisation
    - for each particle in the swarm, initialise the position from a uniform
      distribution with b_l and b_u (tbc)
    - find global best position while initialising the swarm
    - sample the velocity per particle from a uniform distribution

2) track prior global best (??)

3) swarm evolution
    - for each particle in the swarm:
        - sample the r_p, r_g values from uniform distribution(0,1)
        - with w = phi_p = phi_g = 0.5, calculate the new velocity with:

          v = w * v + phi_p * r_p * (best_known - curr_pos) + phi_g * r_g * (glob_best - curr_pos)

          *NOTE: consider tuning these scaling values

        - update position
        - if fitness new position better than fitness of best position,
            - update best position
            - if best pos better than global best,
                - update global best
                - check <termination conditions>
    - update generation
    - update prev best

4) termination conditions
    - global best displacement smaller than delta
    - fitness value of best increased by less than threshold epsilon
    - exceeded max generations

5) return global best

"""
import operator
import random

import numpy

from deap import base
from deap import benchmarks
from deap import creator
from deap import tools

GMAX = 100

# Creates a fitness object that minimises its fitness value
creator.create("FitnessMin", base.Fitness, weights=(-1.0,))

# check all the attributes that we will need !!
# Creates a particle with initial declaration of its contained attributes
# Particle creator is a list container that holds the attributes: fitness, speed, .... NOTE: update

creator.create("Particle", list, fitness=creator.FitnessMin, speed=list, best_known=None)

# ----------------------------Toolbox Functions------------------------------
# evaluates the fitness of the position
def evalparabola(individual):
    return individual[0]**2 + individual[1]**2,

# generates and returns a particle based on the dim (size) of the problem
def generate(size, bound_l, bound_u):
    particle = creator.Particle(random.uniform(bound_l,bound_u) for _ in range(size))
    bound = bound_u - bound_l
    particle.speed = [random.uniform(-abs(bound), abs(bound)) for _ in range(size)]
    particle.best_known = creator.Particle(particle)
    return particle

def updateParticle(particle):
    

toolbox = base.Toolbox()
toolbox.register("evaluate", evalparabola)
toolbox.register("particle", generate, size=2, bound_l=-10, bound_u=10)
toolbox.register("population", tools.initRepeat, list, toolbox.particle)
# register update function

# -----------------------------Main Algorithm--------------------------------
def main():
    pop = toolbox.population(n=50)
    g = 1
    best = None

    for particle in pop:
        particle.fitness.values = toolbox.evaluate(particle)
        # NOTE: when comparing minimised fitness values, a > b returns true if
        # a is smaller than b
        if not best or best.fitness.values > particle.fitness.values:
            best = creator.Particle(particle)
            best.fitness.values = particle.fitness.values

    while g <= GMAX:
        for particle in pop:
            r_p = random.uniform(0,1)
            r_g = random.uniform(0,1)
            toolbox.update(particle)



if __name__ == "__main__":
    main()
