"""
Particle Swarm Optimisation with improvements - Jie Jenny Yan, January 2018
- Framework: DEAP, Scoop
- Fitness function: Optimisation Test Functions
- Particle Attributes: velocity, best_known, curr_pos
- Constants: upper/lower bounds (b_u, b_l: generate function, random.uniform),
             inertia weighting (w: swarm evolution function -> velocity equation),
             accel coefficients (phi_p, phi_g: swarm evolution function -> velocity equation),
             diversify search (r_p, r_g: swarm evolution function, velocity equation),

PSO Algorithm:

1) swarm initialisation
    - for each particle in the swarm, initialise the position from a uniform
      distribution with b_l and b_u (tbc)
    - find global best position while initialising the swarm
    - sample the velocity per particle from a uniform distribution

2) track prior global best (for terminating condition)

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
import numpy as np

from deap import base
from deap import benchmark
from deap import creator
from deap import tools

from scoop import futures

GMAX = 500             # Max number of generations
DELTA = 1e-7           # Smallest position increment allowed
EPSILON = 1e-7         # Smallest fitness value increment allowed
DIM = 4                # No. of Dimensions in the problem
POPULATION = 50        # Size of the particle swarm

# ----------------------------Optimisation Functions------------------------------
# Evaluates the fitness of the position

# NOTE: need to check whether the particles will still find the right place`
# if they start at a point far away from the optima

def sphere(individual):
    return -1.0 * np.sum(individual - 3.)**2,

def rastrigin(individual):
    # 0.8,0.8,0.8
    sq_component = individual **2
    cos_component = np.cos(2 * np.pi * individual)
    summation = np.subtract(sq_component, 10 * cos_component)
    return -1.0 * (10 * DIM + np.add.reduce(summation)),

def ackley(individual):
    sqrt_component = np.sqrt(0.5 * np.add.reduce(np.square(individual)))
    # NOTE: converges to -1, 2 dimensions only
    cos_component = 0.5 * np.cos(2 * np.pi * individual)
    return -1.0 * (-20 * np.exp(-0.2 * sqrt_component) - np.exp(cos_component) + np.exp(1) + 20)

def rosenbrock(individual):
    summation = np.array([100*((individual[i+1] - individual[i]**2)**2) + (individual[i] - 1)**2 for i in range(len(individual)-2)])
    return -np.add.reduce(summation),

def beale(individual):
    #NOTE: 2 dimensions only
    x = individual[0]
    y = individual[1]
    first = 1.5 - x + x * y
    second = 2.25 - x + x * y ** 2
    third = 2.625 - x + x * y ** 3
    return -1.0 * (first ** 2 + second ** 2 + third ** 2),

def bukin6(individual):
    #NOTE:-15 <= x <= -5, -3 <= y <= 3
    x = individual[0]
    y = individual[1]
    sqrt_component = np.sqrt(abs(y - 0.01 * x **2))
    abs_component = 0.01 * abs(x + 10)
    return -1.0 * (100 * sqrt_component + abs_component),

# --------------------------Swarm operations ---------------------------------

# generates and returns a particle based on the dim (size) of the problem
def generate(bound_l, bound_u):
    particle = creator.Particle(np.random.uniform(bound_l,bound_u) for _ in range(DIM))
    bound = bound_u - bound_l
    particle.velocity = np.array([np.random.uniform(-abs(bound), abs(bound)) for _ in range(DIM)])
    particle.best_known = creator.Particle(particle)
    return particle

# updating the velocity and position of the particle
def updateParticle(particle, best, w, phi_p, phi_g):
    r_p = np.array([np.random.uniform(0,1) for _ in particle])
    r_g = np.array([np.random.uniform(0,1) for _ in particle])

    p = np.subtract(particle.best_known, particle)
    g = np.subtract(best, particle)

    v_p = phi_p * np.multiply(p, r_p)
    v_g = phi_g * np.multiply(g, r_g)

    v_w = w * particle.velocity
    particle.velocity = np.add(v_w, np.add(v_p, v_g))
    particle[:] = np.add(particle, particle.velocity)

# initialise the swarm with fitness values.
def initialiseSwarm(particle, best):

    # assigning the fitness values and initialising best known position
    particle.fitness.values = toolbox.evaluate(particle)
    particle.best_known.fitness.values = particle.fitness.values

    # NOTE: when comparing minimised fitness values, a > b returns true if
    # a is smaller than b
    if best is None or (best.fitness.values > particle.fitness.values):

        best = creator.Particle(particle)
        best.fitness.values = particle.fitness.values

    return best

# updates a particle with new values
def updateSwarm(particle, best):

    # move the particles with the update function and eval new fitness
    toolbox.update(particle, best)
    particle.fitness.values = toolbox.evaluate(particle)

    # update the best_known position
    if particle.fitness.values > particle.best_known.fitness.values:

        particle.best_known = creator.Particle(particle)
        particle.best_known.fitness.values = particle.fitness.values

# ---------------------- toolbox -------------------------------------
# registering all the functions to the toolbox
toolbox = base.Toolbox()
toolbox.register("evaluate", rastrigin)
toolbox.register("particle", generate, bound_l=-5, bound_u=5)
toolbox.register("population", tools.initRepeat, list, toolbox.particle)
toolbox.register("update", updateParticle, phi_p=0.8, phi_g=0.8, w=0.8)
toolbox.register("map", futures.map)

# ---------------------------------------------------------------------

def main():
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", np.mean)
    stats.register("std", np.std)
    stats.register("min", np.min)
    stats.register("max", np.max)

    logbook = tools.Logbook()
    logbook.header = ["gen"] + stats.fields

    pop = toolbox.population(POPULATION)
    #NOTE: consider a better way to do best 
    best = list(map(initialiseSwarm, pop))
