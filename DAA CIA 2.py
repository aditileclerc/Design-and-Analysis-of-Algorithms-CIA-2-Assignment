#!/usr/bin/env python
# coding: utf-8

# In[1]:


#3

#importing packages
import numpy as np
import matplotlib.pyplot as plt

#Init the global variables
LOWER = -1    ## lower limit of the function
UPPER = 2     ## Upper limit of the function
SIZE = 10     ## Size of the population

#Function to be profiled
def function(x):
    return x * np.sin(10 * np.pi * x) + 2

#Generating the intial / first generation of the population
##number of bits to represent - resolution
def generate_population(size = SIZE, resolution = 7):
    # This function will generate the population in terms of binary
    population = []
    for _ in range(size):
        population += [np.array([np.random.randint(1000) % 2 for _ in range(resolution)])]
    return np.array(population)
#Decoding the individual in a population
def decode(individual):
    convertor = np.array([2**(len(individual) - 1 -i) for i in range(len(individual))])
    ##[2**6,2**5,2**4,2**3,2**2,2**1,2**0] . [1,0,1,0,1,1,1]
    number = np.dot(convertor,individual) / 2**len(individual)
    min = np.dot(convertor,np.array([0 for i in range(len(individual))])) / 2**len(individual) ##[0,0,0,0,0,0,0]
    max = np.dot(convertor,np.array([1 for i in range(len(individual))])) / 2**len(individual) ##[1,1,1,1,1,1,1]
   
    #Scale to the function range
    final = (UPPER - LOWER) * (number - min) / (max - min) + LOWER
    return final
#Fitness Function
def fitnessFunc(individual):
    value = decode(individual)
    y = function(value)
    return y
#Mutation Function
def mutate(individual,mutation_rate = 0.01):
    #bit-flip mut
    if(np.random.random() > (1-mutation_rate)):
        index = np.random.randint(len(individual))
        individual[index] = not individual[index]
       
    return individual
#Reproduction - Crossover + Mutation Function - Will produce 4 children
def reproduce(individual1,individual2):
    #SPC
    split_point = np.random.randint(len(individual1))
   
    child_1 = np.concatenate((individual1[:split_point] , individual2[split_point:]))
    child_2 = np.concatenate((individual2[:split_point] , individual1[split_point:]))
   
    child_3 = np.concatenate((individual1[split_point:] , individual2[:split_point]))
    child_4 = np.concatenate((individual2[split_point:] , individual1[:split_point]))
   
    child_1 = mutate(child_1)
    child_2 = mutate(child_2)
    child_3 = mutate(child_3)
    child_4 = mutate(child_4)
   
    return [child_1,child_2,child_3,child_4]
#Adding children/New individuals into the population while checking for duplicate entries
def append_children(population,children):
    for child in children:
        if child.tolist() not in population.tolist():
            population = np.concatenate((population,np.array([child])))
           
    return population
#Formation of the next generation
def form_Next_Population(population):
    pop_size = len(population)
   
    for ind1_index in range(pop_size):
        for ind2_index in range(pop_size):
            if ind1_index != ind2_index:
                children = reproduce(population[ind1_index],population[ind2_index])
                population = append_children(population, children)
               
    fitness_value = []
    for individual in population:
        fitness_value += [fitnessFunc(individual)]
    sorted_population = np.argsort(fitness_value)[::-1][:SIZE]
   
    return population[sorted_population]

#%%

##The "Question" Plot
x = [-1 + (i * 3/2**7) for i in range(2**7)]
y = [function(i) for i in x]

##Calling the first generation population
POP = generate_population()
##Keeps track of population
generation = 0
#%%

##To Plot the Indiviual in the plot
individuals = []
functionValue = []
for individual in POP:
    individuals += [decode(individual)]    ##X Values
    functionValue += [function(decode(individual))] ##Y Values
   
##Sorting the points so that the profile lines are neat and not haphazard
temp = np.argsort(individuals)
individuals = np.array(individuals)[temp]
functionValue = np.array(functionValue)[temp]
   
plt.title(generation)
plt.plot(x,y)
plt.scatter(individuals,functionValue)
plt.plot(individuals,functionValue)

##Call the function that forms the next generation
POP = form_Next_Population(POP)
generation += 1
   



# In[2]:


#1

# Python3 implementation of the above approach
from random import randint

INT_MAX = 2147483647
# Number of cities in TSP
V = 5

# Names of the cities
GENES = "ABCDE"

# Starting Node Value
START = 0

# Initial population size for the algorithm
POP_SIZE = 10

# Structure of a GNOME
# defines the path traversed
# by the salesman while the fitness value
# of the path is stored in an integer


class individual:
	def __init__(self) -> None:
		self.gnome = ""
		self.fitness = 0

	def __lt__(self, other):
		return self.fitness < other.fitness

	def __gt__(self, other):
		return self.fitness > other.fitness


# Function to return a random number
# from start and end
def rand_num(start, end):
	return randint(start, end-1)


# Function to check if the character
# has already occurred in the string
def repeat(s, ch):
	for i in range(len(s)):
		if s[i] == ch:
			return True

	return False


# Function to return a mutated GNOME
# Mutated GNOME is a string
# with a random interchange
# of two genes to create variation in species
def mutatedGene(gnome):
	gnome = list(gnome)
	while True:
		r = rand_num(1, V)
		r1 = rand_num(1, V)
		if r1 != r:
			temp = gnome[r]
			gnome[r] = gnome[r1]
			gnome[r1] = temp
			break
	return ''.join(gnome)


# Function to return a valid GNOME string
# required to create the population
def create_gnome():
	gnome = "0"
	while True:
		if len(gnome) == V:
			gnome += gnome[0]
			break

		temp = rand_num(1, V)
		if not repeat(gnome, chr(temp + 48)):
			gnome += chr(temp + 48)

	return gnome


# Function to return the fitness value of a gnome.
# The fitness value is the path length
# of the path represented by the GNOME.
def cal_fitness(gnome):
	mp = [
		[0, 2, INT_MAX, 12, 5],
		[2, 0, 4, 8, INT_MAX],
		[INT_MAX, 4, 0, 3, 3],
		[12, 8, 3, 0, 10],
		[5, INT_MAX, 3, 10, 0],
	]
	f = 0
	for i in range(len(gnome) - 1):
		if mp[ord(gnome[i]) - 48][ord(gnome[i + 1]) - 48] == INT_MAX:
			return INT_MAX
		f += mp[ord(gnome[i]) - 48][ord(gnome[i + 1]) - 48]

	return f


# Function to return the updated value
# of the cooling element.
def cooldown(temp):
	return (90 * temp) / 100


# Comparator for GNOME struct.
# def lessthan(individual t1,
#			 individual t2)
# :
#	 return t1.fitness < t2.fitness


# Utility function for TSP problem.
def TSPUtil(mp):
	# Generation Number
	gen = 1
	# Number of Gene Iterations
	gen_thres = 5

	population = []
	temp = individual()

	# Populating the GNOME pool.
	for i in range(POP_SIZE):
		temp.gnome = create_gnome()
		temp.fitness = cal_fitness(temp.gnome)
		population.append(temp)

	print("\nInitial population: \nGNOME	 FITNESS VALUE\n")
	for i in range(POP_SIZE):
		print(population[i].gnome, population[i].fitness)
	print()

	found = False
	temperature = 10000

	# Iteration to perform
	# population crossing and gene mutation.
	while temperature > 1000 and gen <= gen_thres:
		population.sort()
		print("\nCurrent temp: ", temperature)
		new_population = []

		for i in range(POP_SIZE):
			p1 = population[i]

			while True:
				new_g = mutatedGene(p1.gnome)
				new_gnome = individual()
				new_gnome.gnome = new_g
				new_gnome.fitness = cal_fitness(new_gnome.gnome)

				if new_gnome.fitness <= population[i].fitness:
					new_population.append(new_gnome)
					break

				else:

					# Accepting the rejected children at
					# a possible probability above threshold.
					prob = pow(
						2.7,
						-1
						* (
							(float)(new_gnome.fitness - population[i].fitness)
							/ temperature
						),
					)
					if prob > 0.5:
						new_population.append(new_gnome)
						break

		temperature = cooldown(temperature)
		population = new_population
		print("Generation", gen)
		print("GNOME	 FITNESS VALUE")

		for i in range(POP_SIZE):
			print(population[i].gnome, population[i].fitness)
		gen += 1


if __name__ == "__main__":

	mp = [
		[0, 2, INT_MAX, 12, 5],
		[2, 0, 4, 8, INT_MAX],
		[INT_MAX, 4, 0, 3, 3],
		[12, 8, 3, 0, 10],
		[5, INT_MAX, 3, 10, 0],
	]
	TSPUtil(mp)


# In[7]:


#5

import random as rn
import numpy as np
from numpy.random import choice as np_choice

class AntColony(object):

    def __init__(self, distances, n_ants, n_best, n_iterations, decay, alpha=1, beta=1):
        """
        Args:
            distances (2D numpy.array): Square matrix of distances. Diagonal is assumed to be np.inf.
            n_ants (int): Number of ants running per iteration
            n_best (int): Number of best ants who deposit pheromone
            n_iteration (int): Number of iterations
            decay (float): Rate it which pheromone decays. The pheromone value is multiplied by decay, so 0.95 will lead to decay, 0.5 to much faster decay.
            alpha (int or float): exponenet on pheromone, higher alpha gives pheromone more weight. Default=1
            beta (int or float): exponent on distance, higher beta give distance more weight. Default=1
        Example:
            ant_colony = AntColony(german_distances, 100, 20, 2000, 0.95, alpha=1, beta=2)          
        """
        self.distances  = distances
        self.pheromone = np.ones(self.distances.shape) / len(distances)
        self.all_inds = range(len(distances))
        self.n_ants = n_ants
        self.n_best = n_best
        self.n_iterations = n_iterations
        self.decay = decay
        self.alpha = alpha
        self.beta = beta

    def run(self):
        shortest_path = None
        all_time_shortest_path = ("placeholder", np.inf)
        for i in range(self.n_iterations):
            all_paths = self.gen_all_paths()
            self.spread_pheronome(all_paths, self.n_best, shortest_path=shortest_path)
            shortest_path = min(all_paths, key=lambda x: x[1])
            print (shortest_path)
            if shortest_path[1] < all_time_shortest_path[1]:
                all_time_shortest_path = shortest_path            
            self.pheromone = self.pheromone * self.decay            
        return all_time_shortest_path

    def spread_pheronome(self, all_paths, n_best, shortest_path):
        sorted_paths = sorted(all_paths, key=lambda x: x[1])
        for path, dist in sorted_paths[:n_best]:
            for move in path:
                self.pheromone[move] += 1.0 / self.distances[move]

    def gen_path_dist(self, path):
        total_dist = 0
        for ele in path:
            total_dist += self.distances[ele]
        return total_dist

    def gen_all_paths(self):
        all_paths = []
        for i in range(self.n_ants):
            path = self.gen_path(0)
            all_paths.append((path, self.gen_path_dist(path)))
        return all_paths

    def gen_path(self, start):
        path = []
        visited = set()
        visited.add(start)
        prev = start
        for i in range(len(self.distances) - 1):
            move = self.pick_move(self.pheromone[prev], self.distances[prev], visited)
            path.append((prev, move))
            prev = move
            visited.add(move)
        path.append((prev, start)) # going back to where we started    
        return path

    def pick_move(self, pheromone, dist, visited):
        pheromone = np.copy(pheromone)
        pheromone[list(visited)] = 0

        row = pheromone ** self.alpha * (( 1.0 / dist) ** self.beta)

        norm_row = row / row.sum()
        move = np_choice(self.all_inds, 1, p=norm_row)[0]
        return move


# In[8]:


pip install PySwarms


# In[12]:


#4

# Import PySwarms
import pyswarms as ps
from pyswarms.utils.functions import single_obj as fx
# Set-up hyperparameters
options = {'c1': 0.5, 'c2': 0.3, 'w':0.9}
# Call instance of PSO
optimizer = ps.single.GlobalBestPSO(n_particles=10, dimensions=2, options=options)
# Perform optimization
cost, pos = optimizer.optimize(fx.sphere, iters=1000)
import matplotlib.pyplot as plt
from pyswarms.utils.plotters import plot_contour, plot_surface
from pyswarms.utils.plotters.formatters import Designer
from pyswarms.utils.plotters.formatters import Mesher
m = Mesher(func=fx.sphere)
# Make animation
animation = plot_contour(pos_history=optimizer.pos_history,
                         mesher=m,
                         mark=(0,0)) # Mark minima


# In[ ]:


#4

import math
import random

# Define the bounds of the search space
xmin = -10
xmax = 10

# Define the parameters of the PSO algorithm
num_particles = 20
max_iterations = 100
c1 = 2.0
c2 = 2.0
w = 0.7

# Define the fitness function (sine function)
def fitness(x):
    return math.sin(x)

# Initialize the particles with random positions and velocities
particles = []
for i in range(num_particles):
    position = random.uniform(xmin, xmax)
    velocity = random.uniform(xmin, xmax)
    particles.append({'position': position, 'velocity': velocity, 'best_position': position, 'best_fitness': fitness(position)})

# Start the iterations
global_best_position = float('-inf')
global_best_fitness = float('-inf')
for iteration in range(max_iterations):
    # Update the velocity and position of each particle
    for i in range(num_particles):
        particle = particles[i]
        particle['velocity'] = w * particle['velocity'] + c1 * random.random() * (particle['best_position'] - particle['position']) + c2 * random.random() * (global_best_position - particle['position'])
        particle['position'] += particle['velocity']
        if particle['position'] < xmin:
            particle['position'] = xmin
            particle['velocity'] = 0
        if particle['position'] > xmax:
            particle['position'] = xmax
            particle['velocity'] = 0
        particle_fitness = fitness(particle['position'])
        if particle_fitness > particle['best_fitness']:
            particle['best_position'] = particle['position']
            particle['best_fitness'] = particle_fitness
        if particle_fitness > global_best_fitness:
            global_best_position = particle['position']
            global_best_fitness = particle_fitness

# Print the results
print('Global best position: ', global_best_position)
print('Global best fitness: ', global_best_fitness)


# In[ ]:


#2

import random

# Define the TSP problem instance (city coordinates)
cities = [(60, 200), (180, 200), (80, 180), (140, 180), (20, 160), (100, 160), (200, 160), (140, 140), (40, 120), (100, 120), (180, 100), (60, 80), (120, 80), (180, 60), (20, 40), (100, 40), (200, 40), (20, 20), (60, 20), (160, 20)]

# Define the size of the population
pop_size = 20

# Define the number of iterations
num_iterations = 100

# Define the number of beliefs
num_beliefs = 5

# Define the probability of innovation
p_innovate = 0.1

# Define the number of trials
num_trials = 10

# Define the fitness function
def fitness(individual):
    distance = 0
    for i in range(len(individual) - 1):
        city_a = cities[individual[i]]
        city_b = cities[individual[i+1]]
        distance += ((city_a[0] - city_b[0]) ** 2 + (city_a[1] - city_b[1]) ** 2) ** 0.5
    return distance

# Initialize the population
population = []
for i in range(pop_size):
    individual = list(range(len(cities)))
    random.shuffle(individual)
    population.append(individual)

# Start the iterations
for iteration in range(num_iterations):
    # Sort the population by fitness
    population.sort(key=fitness)

    # Select the beliefs
    beliefs = []
    for i in range(num_beliefs):
        beliefs.append(population[i])

    # Generate new solutions
    for i in range(pop_size):
        if random.random() < p_innovate:
            individual = list(range(len(cities)))
            random.shuffle(individual)
            population[i] = individual
        else:
            trial = []
            for j in range(len(cities)):
                belief = random.choice(beliefs)
                trial.append(belief[j])
            population[i] = trial

    # Evaluate the new population
    for i in range(pop_size):
        fitness_i = fitness(population[i])
        for j in range(num_trials):
            trial = []
            for k in range(len(cities)):
                belief = random.choice(beliefs)
                trial.append(belief[k])
            fitness_j = fitness(trial)
            if fitness_j < fitness_i:
                population[i] = trial
                fitness_i = fitness_j

# Print the best solution
best_individual = min(population, key=fitness)
print('Best individual: ', best_individual)
print('Fitness: ', fitness(best_individual))


# In[ ]:




