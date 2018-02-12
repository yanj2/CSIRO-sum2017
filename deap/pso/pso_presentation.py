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
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm

from deap import base
from deap import benchmarks
from deap import creator
from deap import tools

GMAX = 500
DELTA = 1e-7
EPSILON = 1e-7
DIM = 2
PARTICLE_MIN = -5
PARTICLE_MAX = 5
STEP_SIZE = 0.01

# ------------------------surface plot ------------------------------------


def swarm_plotter(axes,
                  swarm,
                  fitness_fn,
                  xmin=PARTICLE_MIN,
                  xmax=PARTICLE_MAX,
                  ymin=PARTICLE_MIN,
                  ymax=PARTICLE_MAX,
                  stepsize=0.025):

    x = np.arange(xmin, xmax, stepsize)
    y = np.arange(ymin, ymax, stepsize)
    X, Y = np.meshgrid(x, y)
    Z = np.zeros_like(X)

    for i in range(len(x)):
        for j in range(len(y)):
            Z[i, j] = fitness_fn([X[i, j], Y[i, j]])[0]

    # axes.contour(X, Y, Z)
    surf = plt.imshow(-Z,
               interpolation='bilinear',
               origin='lower',
               #cmap=cm.reversed(bone),
               extent=(xmin, xmax, ymin, ymax))

    x_list = []
    y_list = []

    for particle in swarm:
        x_list.append(particle[0])
        y_list.append(particle[1])

    axes.scatter(x_list, y_list, c='w')
    axes.plot([3],[0.5],'ro')
    axes.set_xlim(xmin, xmax)
    axes.set_ylim(ymin, ymax)
    return surf 

    	
# -------------------------------------------------------------------------
# Creates a fitness object that minimises its fitness value
creator.create("Fitness", base.Fitness, weights=(1.0,))
creator.create("Particle", np.ndarray, fitness=creator.Fitness, velocity=np.ndarray(DIM), best_known=None)

# ----------------------------Optimisation Functions------------------------------
# evaluates the fitness of the position

# NOTE: need to check whether the particles will still find the right place`
# if they start at a point far away from the optima

def sphere(individual):
    return -1.0 * np.sum(np.array(individual)**2),

def rastrigin(individual):
    # 0.8,0.8,0.8
    sq_component = np.array(individual) **2
    cos_component = np.cos(2 * np.pi * np.array(individual))
    summation = np.subtract(sq_component, 10 * cos_component)
    return -1.0 * (10 * DIM + np.add.reduce(summation)),

def ackley(individual):
    sqrt_component = np.sqrt(0.5 * np.add.reduce(np.square(np.array(individual))))
    # NOTE: converges to -1, 2 dimensions only
    cos_component = 0.5 * np.cos(2 * np.pi * np.array(individual))
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
def generate(size, bound_l, bound_u):
    particle = creator.Particle(np.random.uniform(bound_l,bound_u) for _ in range(size))
    bound = bound_u - bound_l
    particle.velocity = np.array([np.random.uniform(-abs(bound), abs(bound)) for _ in range(size)])
    particle.best_known = creator.Particle(particle)
    return particle

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

# registering all the functions to the toolbox
toolbox = base.Toolbox()
toolbox.register("evaluate", beale)
toolbox.register("particle", generate, size=DIM, bound_l=-5, bound_u=0)
toolbox.register("population", tools.initRepeat, list, toolbox.particle)
toolbox.register("update", updateParticle, phi_p=0.8, phi_g=0.8, w=0.8)

# -----------------------------Main Algorithm--------------------------------
def main():

    # initialising our population and stats
    pop = toolbox.population(n=50)
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

        if best is None or (best.fitness.values > particle.fitness.values):

            best = creator.Particle(particle)
            best.fitness.values = particle.fitness.values

    figure = plt.figure()
    axes = plt.gca()
    surf = swarm_plotter(axes,
                  pop,
                  toolbox.evaluate,
                  xmin=PARTICLE_MIN,
                  xmax=PARTICLE_MAX,
                  ymin=PARTICLE_MIN,
                  ymax=PARTICLE_MAX,
                  stepsize=STEP_SIZE)

    figure.colorbar(surf, shrink=0.5, aspect=5)
    plt.savefig("beale00.png")
    plt.cla()

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

        swarm_plotter(axes,
                      pop,
                      toolbox.evaluate,
                      xmin=PARTICLE_MIN,
                      xmax=PARTICLE_MAX,
                      ymin=PARTICLE_MIN,
                      ymax=PARTICLE_MAX,
                      stepsize=STEP_SIZE)

        plt.savefig("beale{0:0>2}.png".format(g))
        plt.cla()

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
