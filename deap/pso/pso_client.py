"""
Particle Swarm Optimisation with improvements: BBComp Client - Jie Jenny Yan, January 2018
- Framework: DEAP
- Fitness function: Bbcomp functions
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
import numpy as np

from ctypes import *
from numpy.ctypeslib import ndpointer
import sys
import platform
import json

from deap import base
from deap import benchmarks
from deap import creator
from deap import tools

# get library name
dllname = ""
if platform.system() == "Windows":
	dllname = "./bbcomp.dll"
elif platform.system() == "Linux":
	dllname = "./libbbcomp.so"
elif platform.system() == "Darwin":
	dllname = "./libbbcomp.dylib"
else:
	sys.exit("unknown platform")

# initialize dynamic library
bbcomp = CDLL(dllname)
bbcomp.configure.restype = c_int
bbcomp.login.restype = c_int
bbcomp.numberOfTracks.restype = c_int
bbcomp.trackName.restype = c_char_p
bbcomp.setTrack.restype = c_int
bbcomp.numberOfProblems.restype = c_int
bbcomp.setProblem.restype = c_int
bbcomp.dimension.restype = c_int
bbcomp.numberOfObjectives.restype = c_int
bbcomp.budget.restype = c_int
bbcomp.evaluations.restype = c_int
bbcomp.evaluate.restype = c_int
bbcomp.evaluate.argtypes = [ndpointer(c_double, flags="C_CONTIGUOUS"), ndpointer(c_double, flags="C_CONTIGUOUS")]
bbcomp.history.restype = c_int
bbcomp.history.argtypes = [c_int, ndpointer(c_double, flags="C_CONTIGUOUS"), ndpointer(c_double, flags="C_CONTIGUOUS")]
bbcomp.errorMessage.restype = c_char_p


print("--------------------------------------")
print("black box competition client in Python")
print("--------------------------------------")
print("")

# change the track name to trialMO for multi-objective optimization
TRACK = "trial"
track = "trialMO"

# set configuration options (this is optional)
result = bbcomp.configure(1, "logs/".encode('ascii'))
if result == 0:
	sys.exit("configure() failed: " + str(bbcomp.errorMessage()))

# login with demo account - this should grant access to the "trial" and "trialMO" tracks (for testing and debugging)
result = bbcomp.login("jieyan".encode('ascii'), "jieyan398".encode('ascii'))
if result == 0:
	sys.exit("login() failed: " + str(bbcomp.errorMessage().decode("ascii")))

print("login successful")

# request the tracks available to this user (this is optional)
numTracks = bbcomp.numberOfTracks()
if numTracks == 0:
	sys.exit("numberOfTracks() failed: " + str(bbcomp.errorMessage().decode("ascii")))

print(str(numTracks) + " track(s):")
for i in range(numTracks):
	trackname = bbcomp.trackName(i).decode("ascii")
	if bool(trackname) == False:
		sys.exit("trackName() failed: " + str(bbcomp.errorMessage().decode("ascii")))

	print("  " + str(i) + ": " + trackname)

# set the track
result = bbcomp.setTrack(track.encode('ascii'))
if result == 0:
	sys.exit("setTrack() failed: " + str(bbcomp.errorMessage().decode("ascii")))

print("track set to " + track)

# obtain number of problems in the track
numProblems = bbcomp.numberOfProblems()
if numProblems == 0:
	sys.exit("numberOfProblems() failed: " << str(bbcomp.errorMessage().decode("ascii")))

print("The track consists of " + str(numProblems) + " problems.")

problemID = 4
result = bbcomp.setProblem(problemID)
if result == 0:
	sys.exit("setProblem() failed: " + str(bbcomp.errorMessage().decode("ascii")))

print("Problem ID set to " + str(problemID))

# obtain problem properties
dim = bbcomp.dimension()
if dim == 0:
	sys.exit("dimension() failed: " + str(bbcomp.errorMessage().decode("ascii")))

obj = bbcomp.numberOfObjectives()
if obj == 0:
	sys.exit("numberOfObjectives() failed: " + str(bbcomp.errorMessage().decode("ascii")))

bud = bbcomp.budget()
if bud == 0:
	sys.exit("budget() failed: " + str(bbcomp.errorMessage().decode("ascii")))

evals = bbcomp.evaluations()
if evals < 0:
	sys.exit("evaluations() failed: " + str(bbcomp.errorMessage().decode("ascii")))

print("problem dimension: " + str(dim))
print("number of objectives: " + str(obj))
print("problem budget: " + str(bud))
print("number of already used up evaluations: " + str(evals))

# ----------------------------Toolbox Functions------------------------------

GMAX = 100
DELTA = 1e-7
EPSILON = 1e-7
BOUND_U = 1
BOUND_L = 0

def evalBbcomp(individual):

    global evals
    # initialise the values
    individual.fitness.values = np.zeros(obj)
    values = np.array(individual.fitness.values)
    points = np.array(individual)
    if (evals >= bud):
    	return values

    # evaluate the points and the values (array form)
    result = bbcomp.evaluate(points, values)
    # check if evaluation was successful
    if result == 0:
    	sys.exit("evaluate() failed: " + str(bbcomp.errorMessage().decode("ascii")))

    evals += 1

    # assign new value if evaluation was successful (tuple form)
    individual.fitness.values = values
    # print for checking
    print("[{}],{},{}".format(evals, individual, individual.fitness.values))
    return values

# generates and returns a particle based on the dim (size) of the problem
def generate(size, bound_l, bound_u):
    particle = creator.Particle(np.random.uniform(bound_l,bound_u) for _ in range(size))
    bound = bound_u - bound_l
    particle.velocity = [np.random.uniform(-abs(bound), abs(bound)) for _ in range(size)]
    particle.best_known = creator.Particle(particle)
    return particle

# NOTE: remember to verify the correctness of this funciton
# updating the velocity and position of the particle

def updateParticle(particle, best, w, phi_p, phi_g):
    r_p = [np.random.uniform(0,1) for _ in particle]
    r_g = [np.random.uniform(0,1) for _ in particle]

    # list of best_known - curr
    p = list(map(operator.sub, particle.best_known, particle))

    # list of global_best - curr
    g = list(map(operator.sub, best, particle))

    # scaled impact of best positions
    v_p = [phi_p * r * x for x,r in zip(p,r_p)]
    v_g = [phi_g * r * x for x,r in zip(g,r_g)]

    # scaled velocity
    v_w = [w * x for x in particle.velocity]
    particle.velocity = list(map(operator.add, v_w, map(operator.add, v_p, v_g)))

    for position, velocity in zip(particle, particle.velocity):
        if position + velocity < BOUND_L or position + velocity > BOUND_U:
            velocity = velocity * -1.0

    particle[:] = list(map(operator.add, particle, particle.velocity))

# -----------------------------------------------------------------------

# Creates a fitness object that minimises its fitness value
creator.create("Fitness", base.Fitness, weights=(-1.0,))

# check all the attributes that we will need !!
# Creates a particle with initial declaration of its contained attributes
# Particle creator is a list container that holds the attributes: fitness, velocity, .... NOTE: update

creator.create("Particle", list, fitness=creator.Fitness, velocity=list, best_known=None)

# registering all the functions to the toolbox
toolbox = base.Toolbox()
toolbox.register("evaluate", evalBbcomp)
toolbox.register("particle", generate, size=dim, bound_l=BOUND_L, bound_u=BOUND_U)
toolbox.register("population", tools.initRepeat, list, toolbox.particle)
toolbox.register("update", updateParticle, phi_p=0.05, phi_g=0.05, w=0.05)

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

        # NOTE: when comparing minimised fitness values, a > b returns true if
        # a is smaller than b
        if not best or (best.fitness.values > particle.fitness.values):

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
