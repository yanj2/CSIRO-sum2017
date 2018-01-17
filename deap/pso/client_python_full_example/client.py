
from ctypes import *
from numpy.ctypeslib import ndpointer
from numpy import *
import sys
import platform
import time

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

# configuration
LOGFILEPATH = "logs/"
USERNAME = "demoaccount"
PASSWORD = "demopassword"
TRACKNAME = "trial"
LOGIN_DELAY_SECONDS = 10
LOCK_DELAY_SECONDS  = 60

# network failure resilient functions
def safeLogin():
	while True:
		result = bbcomp.login(USERNAME, PASSWORD)
		if result != 0:
			return
		msg = bbcomp.errorMessage()
		print "WARNING: login failed: ", msg
		time.sleep(LOGIN_DELAY_SECONDS)
		if msg == "already logged in":
			return

def safeSetTrack():
	while True:
		result = bbcomp.setTrack(TRACKNAME)
		if result != 0:
			return
		print "WARNING: setTrack failed: ", bbcomp.errorMessage()
		safeLogin()

def safeGetNumberOfProblems():
	while True:
		result = bbcomp.numberOfProblems()
		if result != 0:
			return result
		print "WARNING: numberOfProblems failed: ", bbcomp.errorMessage()
		safeSetTrack()

def safeSetProblem(problemID):
	while True:
		result = bbcomp.setProblem(problemID)
		if result != 0:
			return
		msg = bbcomp.errorMessage()
		print "WARNING: setProblem failed: ", msg
		if len(msg) >= 22 and msg[0:22] == "failed to acquire lock":
			time.sleep(LOCK_DELAY_SECONDS)
		else:
			safeSetTrack()

def safeEvaluate(problemID, point):
	while True:
		value = zeros(1)
		result = bbcomp.evaluate(point, value)
		if result != 0: return value[0]
		print "WARNING: evaluate failed: ", bbcomp.errorMessage()
		safeSetProblem(problemID)

def safeGetDimension(problemID):
	while True:
		result = bbcomp.dimension()
		if result != 0: return result
		print "WARNING: dimension failed: ", bbcomp.errorMessage()
		safeSetProblem(problemID)

def safeGetBudget(problemID):
	while True:
		result = bbcomp.budget()
		if result != 0: return result
		print "WARNING: budget failed: ", bbcomp.errorMessage()
		safeSetProblem(problemID)

def safeGetEvaluations(problemID):
	while True:
		result = bbcomp.evaluations()
		if result >= 0: return result
		print "WARNING: evaluations failed: " + bbcomp.errorMessage()
		safeSetProblem(problemID)

# predicate for sorting the simplex
def comp(item1, item2):
	if item1["value"] < item2["value"]: return -1
	if item1["value"] > item2["value"]: return +1
	return 0

# simple constraint handling by truncation
def truncate2bounds(vec):
	for i in range(len(vec)):
		if vec[i] < 0.0: vec[i] = 0.0
		elif vec[i] > 1.0: vec[i] = 1.0
	return vec

# Nelder Mead algorithm
def solveProblem(problemID):
	# set the problem
	safeSetProblem(problemID)

	# obtain problem properties
	bud = safeGetBudget(problemID)
	dim = safeGetDimension(problemID)
	evals = safeGetEvaluations(problemID)

	# output status
	if evals == bud:
		print "problem ", problemID, ": already solved"
		return
	elif evals == 0:
		print "problem ", problemID, ": starting from scratch"
	else:
		print "problem ", problemID, ": starting from evaluation ", evals

	# create the initial simplex
	simplex = []
	for j in range(dim + 1):
		point = zeros(dim)
		for i in range(dim):
			if i == j:
				point[i] = 0.8
			else:
				point[i] = 0.2
		simplex.append({"point": point, "value": 1e100})
	simplex_evaluated = 0

	# recover algorithm state from saved evaluations
	for e in range(evals):
		point = zeros(dim)
		value = zeros(1)
		result = bbcomp.history(e, point, value)
		if result == 0:
			# note: this evaluation is lost
			print "WARNING: history failed."
		else:
			if simplex_evaluated <= dim:
				simplex[simplex_evaluated] = {"point": point, "value": value[0]}
				simplex_evaluated += 1
			else:
				simplex = sorted(simplex, comp);
				if value < simplex[dim]["value"]: simplex[dim] = {"point": point, "value": value[0]}

	# optimization loop
	while safeGetEvaluations(problemID) < bud:
		if simplex_evaluated <= dim:
			# evaluate the initial simplex
			simplex[simplex_evaluated]["value"] = safeEvaluate(problemID, simplex[simplex_evaluated]["point"])
			simplex_evaluated += 1
		else:
			# step of the simplex algorithm
			simplex = sorted(simplex, comp);

			# compute centroid
			x0 = zeros(dim)
			for j in range(dim): x0 += simplex[j]["point"]
			x0 /= dim

			# reflection
			xr = {"point:": zeros(dim), "value": 1e100}
			xr["point"] = truncate2bounds(2.0 * x0 - simplex[dim]["point"])
			xr["value"] = safeEvaluate(problemID, xr["point"])
			if simplex[0]["value"] <= xr["value"] and xr["value"] < simplex[dim]["value"]:
				# replace worst point with reflected point
				simplex[dim] = xr
			elif xr["value"] < simplex[0]["value"]:
				if safeGetEvaluations(problemID) >= bud: break

				# expansion
				xe = {"point:": zeros(dim), "value": 1e100}
				xe["point"] = truncate2bounds(3.0 * x0 - 2.0 * simplex[dim]["point"])
				xe["value"] = safeEvaluate(problemID, xe["point"])
				if xe["value"] < xr["value"]:
					# replace worst point with expanded point
					simplex[dim] = xe
				else:
					# replace worst point with reflected point
					simplex[dim] = xr
			else:
				if safeGetEvaluations(problemID) >= bud: break

				# contraction
				xc = {"point:": zeros(dim), "value": 1e100}
				xc["point"] = truncate2bounds(0.5 * x0 + 0.5 * simplex[dim]["point"])
				xc["value"] = safeEvaluate(problemID, xc["point"])
				if xc["value"] < simplex[dim]["value"]:
					# replace worst point with contracted point
					simplex[dim] = xc
				else:
					# reduction
					for j in range(dim):
						if safeGetEvaluations(problemID) >= bud: break
						simplex[j+1]["point"] = truncate2bounds(0.5 * simplex[0]["point"] + 0.5 * simplex[j+1]["point"])
						simplex[j+1]["value"] = safeEvaluate(problemID, simplex[j+1]["point"])


# setup
result = bbcomp.configure(1, "logs/")
if result == 0:
	sys.exit("configure() failed: " + bbcomp.errorMessage())

safeLogin()
safeSetTrack()
n = safeGetNumberOfProblems()

# solve all problems in the track
for i in range(n):
	solveProblem(i)
