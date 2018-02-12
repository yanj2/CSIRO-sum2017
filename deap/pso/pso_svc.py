"""
Particle Swarm Optimisation with improvements - Jie Jenny Yan, January 2018
- Framework: DEAP
- Fitness function: Parabaloid
- Particle Attributes: velocity, best_known, global_best, curr_pos #CHECK
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
import numpy as np

from sklearn.svm import SVC
from sklearn import datasets
from sklearn.model_selection import cross_val_score

from deap import base
from deap import benchmarks
from deap import creator
from deap import tools

GMAX = 100
DELTA = 1e-4
EPSILON = 1e-4
DIM = 1
# ----------------------------Toolbox Functions----------------------------
# Creates a fitness object that minimises its fitness value
creator.create("Fitness", base.Fitness, weights=(1.0,))

# check all the attributes that we will need !!
# Creates a particle with initial declaration of its contained attributes
# Particle creator is a list container that holds the attributes: fitness, velocity, .... NOTE: update
creator.create("Particle", np.ndarray, fitness=creator.Fitness, velocity=np.ndarray(DIM), best_known=None)
digits = datasets.load_digits()
iris = datasets.load_iris()
# ----------------------------Optimisation Functions--------------------------
# evaluates the fitness of the position
def svc_example(params, data, targets):
    clf = SVC(C=params[0])
    scores = cross_val_score(clf, data, targets, cv=3)
    return scores.mean(),
# --------------------------Swarm operations ---------------------------------

# generates and returns a particle based on the dim (size) of the problem
def generate(size, bound_l, bound_u):
    particle = creator.Particle(np.random.uniform(bound_l,bound_u) for _ in range(size))
    bound = bound_u - bound_l
    particle.velocity = np.array([np.random.uniform(-abs(bound), abs(bound)) for _ in range(size)])
    particle.best_known = creator.Particle(particle)
    return particle

# NOTE: remember to verify the correctness of this funciton
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

    for n in range(len(particle)):
        if particle[n] + particle.velocity[n] < 0:
            particle[n] = -1.0 * (particle[n] + particle.velocity[n])
        else:
            particle[n] = particle[n] + particle.velocity[n]

# registering all the functions to the toolbox
toolbox = base.Toolbox()
toolbox.register("evaluate", svc_example, data=digits.data, targets=digits.target)
toolbox.register("particle", generate, size=DIM, bound_l=0, bound_u=20)
toolbox.register("population", tools.initRepeat, list, toolbox.particle)
toolbox.register("update", updateParticle, phi_p=0.8, phi_g=0.8, w=0.8)

# -----------------------------Main Algorithm--------------------------------
def main():

    # initialising our population and stats
    pop = toolbox.population(n=100)
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", np.mean)
    stats.register("std", np.std)
    stats.register("min", np.min)
    stats.register("max", np.max)

    logbook = tools.Logbook()
    logbook.header = ["gen"] + stats.fields

    g = 1
    best = None

    for particle in pop:

        # assigning the fitness values and initialising best known position
        particle.fitness.values = toolbox.evaluate(particle)
        particle.best_known.fitness.values = particle.fitness.values

        # NOTE: when comparing minimised fitness values, a > b returns true if
        # a is smaller than b
        if best is None or (best.fitness.values > particle.fitness.values):

            best = creator.Particle(particle)
            best.fitness.values = particle.fitness.values

    # assigning the previous best particles
    prev_best = creator.Particle(best)
    prev_best.fitness.values = best.fitness.values

    # evolving the particle population
    while g <= GMAX:
        for particle in pop:

            # move the particles with the update function and eval new fitness
            toolbox.update(particle, best)
            particle.fitness.values = toolbox.evaluate(particle)

            # update the best_known position
            if particle.fitness.values > particle.best_known.fitness.values:

                particle.best_known = creator.Particle(particle)
                particle.best_known.fitness.values = particle.fitness.values

                # if relevant update the best global position
                if particle.best_known.fitness.values > best.fitness.values:

                    best = creator.Particle(particle.best_known)
                    best.fitness.values = particle.best_known.fitness.values

                    # if the fitness has converged, stop evolving
                    if best.fitness.values[0] - prev_best.fitness.values[0] < EPSILON:
                        logbook.record(gen=g, **stats.compile(pop))
                        print("fitness values")
                        print(logbook.stream)
                        return pop, best

                    # if the posiiton has converged, stop evolving
                    if np.sqrt(np.add.reduce(np.square(np.subtract(best, prev_best)))) < DELTA:
                        logbook.record(gen=g, **stats.compile(pop))
                        print("position values")
                        print(logbook.stream)
                        return pop, best

        # keep track of the previous best position
        prev_best = creator.Particle(best)
        prev_best.fitness.values = best.fitness.values

        # update our records
        logbook.record(gen=g, **stats.compile(pop))
        g = g + 1

    print(logbook.stream)
    return pop, best

if __name__ == "__main__":
    print(main())
