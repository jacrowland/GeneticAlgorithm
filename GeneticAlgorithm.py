import random
import numpy as np
import matplotlib.pyplot as plt

"""

GeneticAlgorithm.py

Implementation of the standard Genetic algorithm (GA) for binary strings

Information on GA's -> https://en.wikipedia.org/wiki/Genetic_algorithm

Author: Jacob Rowland (2021)

"""
class GeneticAlgorithm():
    """ 

    This GA class iterates a population of individuals by evaluating their fitness at each generation (step).
    The simulation stops when an individual in the current population reaches the threshold fitness value.

    Characteristics include:
    - Ability to for individuals to reproduce themselves (using single-point cross-over)
    - Variety amongst individuals through mutation
    - Evaluation score from a fitness function defined as the number of 1's in an individual hypothesis

    """
    def __init__(self, threshold=15, p=10, r=0.2, m=0.2, k=15):
        """

        Parameters:

        generation(int): the current generation (/ total number of generations)
        population(list[str]): Contains the individual hypothesis binary strings e.g. '010101010'
        threshold(int): Fitness threshold - stop's simulation when an individual in the population reaches this value
        p(int): Number of members in the population
        r(float): Fraction of population to replace by Crossover each iteration
        m(float): Fraction of new population to mutate each iteration
        k(int): length of each individual
        fitnessScores(list[int]): A list of fitness scores for each corresponding population individual e.g. individual at index 0 has their fitness at index 0 in this list
        generationFitness(list[tuple(int, int))]: List of tuples of (generation, maxFitness)
        
        """
        self.generation = 0

        self.population = []
        self.generationalFitness = []
        self.fitnessScores = []

        self.threshold = threshold
        self.p = p
        self.r = r
        self.m = m
        self.k = k

        # Initialise a population of size p with individuals of length k 
        newPopulation = []
        for x in range(self.p):
            newPopulation.append(self.generateHypothesis(self.k))
            self.setPopulation(newPopulation)

        # begin evolving the population
        print(self)
        self.run()

    def run(self):
        """

        Evolves the population until it converges on an individual hypothesis (a solution)
        that is greater than or equal to the threshold fitness value. Essentially 
        simulating a natural selection-esque process

        """
        while max(self.fitnessScores) < self.threshold:
            self.generation += 1
            newPopulation = []
            weights = self.getPopulationWeights()
            newPopulation = self.selectPopulation(weights)
            newPopulation = self.crossoverPopulation(newPopulation, weights)
            newPopulation = self.mutatePopulation(newPopulation)
            while len(newPopulation) > self.p:
                # randomly 'kill' members so the size doesn't exceed self.p
                newPopulation.pop(random.randint(0, len(newPopulation)-1))
            self.setPopulation(newPopulation)
        print(self)
        self.plotFitness()
        return sorted(self.population, key=self.fitness, reverse=True)

    def selectPopulation(self, weights:list)->list[str]:
        """

        This method probabilistically selects (1-r)*p individuals from the current population
        to bring into the next generation (without replacement)

        Paramaters:
        weights(list[floats]): Defines the likelihood a individual will be selected 

        Returns:
        list[str]: A list of selected population member strings

        """
        newPopulation = []
        numToSelect = round((1 - self.r) * len(self.population))
        newPopulation = list(np.random.choice(self.population,numToSelect,replace=False, p=weights))
        return newPopulation

    def crossoverPopulation(self, population:list[str], weights:list)->list[str]:
        """

        Probabilistically selects (without replacement) (r * p) / 2 pairs from the supplied population to perform 
        the genetic operation crossover on. Cross-over generates two 'children' from two 'parent' hypothesis

        Paramaters:
        population (list[str]): A population of individual hypothesis to select from 
        weights (list[int]): Defines the likelihood a individual will be selected

        Returns:
        list[str]: A population of individual hypothesis

        """
        numPairs = round((self.r * len(self.population)) / 2)
        for x in range(numPairs):
            pair = []
            pair = list(np.random.choice(self.population,2,replace=False, p=weights))
            son, daughter = self.crossover(pair[0], pair[1])
            population.append(son)
            population.append(daughter)
        return population

    def mutatePopulation(self, population:list[str])->list[str]:
        """

        Perform a binary point mutation of p * m members of the supplied population

        Paramaters:
        population (list[str]): A list of individual hypothesis strings

        Returns:
        list[str]: A population of individual hypothesis

        """
        numToMutate = round(len(population) * self.m)        
        indices = random.sample(range(0, len(population)), numToMutate)
        for i in indices:
            population[i] = self.mutate(population[i])
        return population

    def getPopulationWeights(self)->list[float]:
        weights = []
        for individual in self.population:
            individualWeight = self.fitness(individual) / sum(self.fitnessScores)
            weights.append(individualWeight)
        return weights

    def setPopulation(self, newPopulation:list[str]):
        """

        Update the current generation population - also triggers the update of the fitness score list and generational list

        Paramaters:
        newPopulation(list[str]): population of individual hypothesis binary strings

        """
        self.population = newPopulation
        self.updatePopulationFitnessScores()
        self.updateGenerationFitnessList()

    def updatePopulationFitnessScores(self):
        self.fitnessScores = self.evaluatePopulationFitness()
    
    def updateGenerationFitnessList(self):
        self.generationalFitness.append((self.generation, max(self.fitnessScores), np.average(self.fitnessScores)))

    def generateHypothesis(self, length:int)->str:
        """

        Generate a new binary hypothesis string

        Parameters:
        length(int): The length of the hypothesis string

        Returns:
        str: A new random individual hypothesis

        """
        hypothesis = []
        for x in range(length):
            hypothesis.append(str(random.choice([0,1])))
        return ''.join(hypothesis)

    def evaluatePopulationFitness(self)->list:
        """

        Evaluate the fitness of the entire population

        Returns:
        list: Fitness scores for each hypothesis in the population

        """
        fitnessScores = []
        for individual in self.population:
            fitnessScores.append(self.fitness(individual))
        return fitnessScores
  
    def fitness(self, individual:str)->int:
        """

        A function that assigns an evaluation score for a hypothesis

        For this simulation it assigns a higher fitness based on the number of 1s in the hypothesis

        Parameters:
        individual(str): An individual hypothesis

        Returns:
        int: The individual hypothesis score

        """
        return individual.count("1")

    def crossover(self, mother:str, father:str):
        """

        Implementation of single-point cross-over.
        
        That is, it takes two population individuals the 'parents' and produces two offspring 'children'. 
        These offspring are the result of splitting the each 'parent' at a random indice
        and crossing over the halves. e.g. splitting 10101010 and 00101010 at the
        midway point results in 10101010 and 00101010

        Parameters:
        mother(str): An individual hypothesis
        father(str): An individual hypothesis

        Return:
        tuple(str, str): A tuple containing two new hypothesis strings

        """
        son = ""
        daughter = ""
        point = random.randrange(0, len(father))
        father1 = father[:point] # first half of father string split at index point
        father2 = father[point:] # second half of father string split at index point
        mother1 = mother[:point]
        mother2 = mother[point:]
        son = father1 + mother2
        daughter = mother1 + father2
        return son, daughter
    
    def mutate(self, individual:str)->str:
        """

        Takes an individual and mutates them by randomly switching one of their bits e.g. a 1 to a 0 or vice versa.

        Parameters:
        individual(str): An individual hypothesis from the population

        Returns:
        individual(str): A mutated individual hypothesis

        """
        i = random.randint(0, len(individual)-1)
        individual = list(individual)
        if individual[i] == "0":
            individual[i] = "1"
        elif individual[i] == "1":
            individual[i] = "0"
        return ''.join(individual)

    def plotFitness(self):
        """

        Display the fitness level over each iteration in a time-series plot

        """
        plt.plot(*zip(*self.generationalFitness))
        plt.legend(['max', 'avg'])
        plt.ylabel('Fitness')
        plt.xlabel('Generation')
        plt.title("Population fitness over time")
        plt.show()
      
    def __str__(self):
        return "GENERATION {} : {}".format(self.generation, sorted(self.population, key=self.fitness, reverse=True))

def main():
    ga = GeneticAlgorithm(threshold=15, p=10, r=0.3, m=0.05, k=15)

if __name__ == "__main__":
    main()