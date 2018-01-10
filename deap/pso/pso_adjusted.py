"""
Particle Swarm Optimisation with improvements - Jie Jenny Yan, January 2018
- Framework: DEAP
- Fitness function: Parabaloid
- Particle Attributes: velocity, curr_best, global_best, curr_pos #CHECK
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

          v = w * v + phi_p * r_p * (curr_best - curr_pos) + phi_g * r_g * (glob_best - curr_pos)

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


# Creates a fitness object that minimises its fitness value
creator.create("FitnessMax", base.Fitness, weights=(-1.0,))

# check all the attributes that we will need !!
# Creates a particle with initial declaration of its contained attributes
# Particle creator is a list container that holds the attributes: fitness, speed, .... NOTE: update
# creator.create("Particle", list, fitness=creator.FitnessMax, speed=None, smin=None, smax=None, best=None)

# ----------------------------Toolbox Functions------------------------------
# evaluates the fitness of the position
def evalparabola(individual):
    return individual[0]**2 + individual[1]**2,
