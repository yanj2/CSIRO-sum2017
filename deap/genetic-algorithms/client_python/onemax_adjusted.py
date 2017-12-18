"""
Using the code originally taken from DEAP onemax example, rewriting for solving
functions in the Black Box Optimisation functions.
"""
from ctypes import *
from numpy.ctypeslib import ndpointer
from numpy import *
import sys
import platform
import json

from deap import base
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
# track = "trialMO"

# set configuration options (this is optional)
result = bbcomp.configure(1, "logs/".encode('ascii'))
if result == 0:
	sys.exit("configure() failed: " + str(bbcomp.errorMessage()))

# login with demo account - this should grant access to the "trial" and "trialMO" tracks (for testing and debugging)
result = bbcomp.login("demoaccount".encode('ascii'), "demopassword".encode('ascii'))
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

# For demonstration purposes we optimize only the first problem in the track.
problemID = 0
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

# -----------------------------------------------------------------------
creator.create("Fitness", base.Fitness, weights=(1.0,))
# Points type for the inputs to the function
creator.create("Points", ndarray)
creator.create("Individual", ndarray, fitness=creator.Fitness)

toolbox = base.Toolbox()
# creates the attribute points to be stored in the individual with `dim` random points
toolbox.register("attr_points", tools.initRepeat, creator.Points, random.rand, dim)

# creates an individual -- consider the '1'...
toolbox.register("individual", tools.initRepeat, creator.Individual,
                 toolbox.attr_points, 1)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

def evalBbcomp(individual):

	global evals

	# initialise the values
	individual.fitness.values = zeros(obj)
	# create a temp array to calculate the values (converts to array form)
	values = array(individual.fitness.values)

	if (evals >= bud):
		return values

	# evaluate the points and the values (array form)
	result = bbcomp.evaluate(individual, values)
	# check if evaluation was successful
	if result == 0:
		sys.exit("evaluate() failed: " + str(bbcomp.errorMessage().decode("ascii")))

	evals += 1

	# assign new value if evaluation was successful (tuple form)
	individual.fitness.values = values
	# print for checking
	print("[{}],{},{}".format(evals, individual, individual.fitness.values))
	return values

toolbox.register("evaluate", evalBbcomp)
toolbox.register("mate", tools.cxOnePoint)
#toolbox.register("mutate", tools.mutFlipBit, indpb=0.05)
# use a different mutation method
toolbox.register("mutate", tools.mutShuffleIndexes, indpb=0.05)
toolbox.register("select", tools.selTournament, tournsize=3)

def main():
    random.seed(64)

    pop = toolbox.population(n=200)

    CXPB, MUTPB = 0.5, 0.2

    print("Start of evolution")

    # Evaluate the entire population
    fitnesses = list(map(toolbox.evaluate, pop))
    for ind, fit in zip(pop, fitnesses):
        ind.fitness.values = fit

    # Extracting all the fitnesses from the individuals
    fits = [ind.fitness.values[0] for ind in pop]

    g = 0
    while (evals < bud):
        g = g + 1
        print("-- Generation %i --" % g)

        # Select the next generation individuals
        offspring = toolbox.select(pop, len(pop))
        offspring = list(map(toolbox.clone, offspring))

        # Apply crossover and mutation on the offspring
        for child1, child2 in zip(offspring[::2], offspring[1::2]):
             if random.random() < CXPB:
                 toolbox.mate(child1[0], child2[0])
                 del child1.fitness.values
                 del child2.fitness.values

        for mutant in offspring:
            if random.random() < MUTPB:
                toolbox.mutate(mutant[0])
                del mutant.fitness.values

        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        fitnesses = map(toolbox.evaluate, invalid_ind)
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit

        pop[:] = offspring

        fits = [ind.fitness.values[0] for ind in pop]

        length = len(pop)
        mean = sum(fits) / length
        sum2 = sum(x*x for x in fits)
        std = abs(sum2 / length - mean**2)**0.5

        print("  Min %s" % min([val for val in fits if val != 0]))
        print("  Max %s" % max(fits))
        print("  Avg %s" % mean)
        print("  Std %s" % std)

    best_ind = tools.selBest(pop, 1)[0]
    print("Best individual is %s, with value %s" % (best_ind, best_ind.fitness.values))

if __name__ == "__main__":

	if evals > 0:
		point = zeros(dim)
		value = zeros(obj)
		for i in range(evals):
			result = bbcomp.history(i, point, value)
			if result == 0:
				sys.exit("history() failed: " + str(bbcomp.errorMessage().decode("ascii")))
			print(point, value)

	main()
	if evals < bud:
		main()

"""
# when using evaluate, since it changes it in place, make a copy of the value
# and evaluate, then adjust
person = toolbox.individual(1)
person.fitness.values = zeros(obj)
print(person.fitness.values)

values = array(person.fitness.values)
print(person[0], values)
result = bbcomp.evaluate(person, values)
print(person[0], values, result)
person.fitness.values = values
print(person[0], person.fitness.values, result)
"""
