import random
import numpy as np
import matplotlib.pyplot as plt

"""

GeneticAlgorithm.py
Author: Jacob Rowland 2021

"""
class GeneticAlgorithm():
    """ 
    
    Implementation of the standard genetic algorithm for binary strings

    This class iterates a population of individuals by evaluating their fitness at each iteration (step).
    The simulation stops when an individual in the current population reaches the threshold fitness value.

    Characteristics include:
    - Ability to for individuals to reproduce themselves (using single-point cross-over)
    - Variety amongst individuals through mutation
    - Evaluation score from Fitness function defined as the number of 1's in an individual hypothesis

    """
    def __init__(self, threshold=15, p=10, r=0.2, m=0.2, length=15):
        """
        Parameters:
        threshold(int): Fitness threshold - stop's simulation when an individual in the population reaches this value
        p(int): Number of members in the population
        r(float): Fraction of population to replace by Crossover each iteration
        m(float): Fraction of new population to mutate each iteration
        length(int): length of each individual
        generation(int): number of populations generate
        generationFitness(tuple[int]): The max fitness for each generation
        """
        self.generation = 0
        self.generationalFitness = []
        self.threshold = threshold
        self.population = []
        self.fitnessScores = []
        self.p = p
        self.r = r
        self.m = m

        for x in range(self.p):
            self.population.append(self.generateHypothesis(length))

        self.run()

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

    def fitness(self, individual:str)->int:
        """
        A function that assigns an evaluation score for a hypothesis

        For this simulation it assigns a higher fitness based on the number of 1s in the hypothesis

        Parameters:
        individual(str): An individual hypothesis

        Returns:
        int: The individual hypothesis score
        """
        # eval score given based on num. of 1s in string
        return individual.count("1")

    def evaluatePopulationFitness(self)->list:
        """
        Evaluate the fitness of the entire population

        Returns:
        list: Fitness scores for each hypothesis in the population
        """
        fitnessScores = []
        for individual in self.population:
            fitnessScores.append(self.fitness(individual))
        self.generationalFitness.append((self.generation, max(fitnessScores), np.average(fitnessScores)))
        return fitnessScores

    def run(self):
        self.fitnessScores = self.evaluatePopulationFitness() # evaluate initial population

        print(self)

        while max(self.fitnessScores) < self.threshold:
            self.generation += 1
            newPopulation = []
            weights = [] # population weights

            # probabilistically select (1-r)*p individuals from the population to bring into the new population
            for individual in self.population:
                individualWeight = self.fitness(individual) / sum(self.fitnessScores)
                weights.append(individualWeight)
            numToSelect = round((1 - self.r) * len(self.population))
            newPopulation = list(np.random.choice(self.population,numToSelect,replace=False, p=weights))

            # cross-over (r * p) / 2 pairs
            numPairs = round((self.r * len(self.population)) / 2)
            for x in range(numPairs):
                pair = []
                pair = list(np.random.choice(self.population,2,replace=False, p=weights))
                son, daughter = self.crossOver(pair[0], pair[1])
                newPopulation.append(son)
                newPopulation.append(daughter)

            # mutate m fraction of the new population
            numToMutate = round(len(newPopulation) * self.m)
            """
            if numToMutate < 1 and numToMutate >= 0:
                numToMutate = 1
            """
            indices = random.sample(range(0, len(newPopulation)), numToMutate)
            for i in indices:
                newPopulation[i] = self.mutate(newPopulation[i])

            # update the population
            self.population = newPopulation

            # evaluate the population
            self.fitnessScores = self.evaluatePopulationFitness()

            print(self)

        self.plotFitness()
        return sorted(self.population, key=self.fitness, reverse=True)

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
        
    def crossOver(self, mother:str, father:str):
        """
        This function implements single-point cross-over. That is, it takes two population 
        individuals and produces two offspring. These offspring are the result of splitting
        the parents at a random indices and crossing over the halves. e.g. splitting 10101010
        and 00101010 at the midway point results in 10101010 and 00101010

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
        Takes an individual and mutates them by randomly switching one of their bits.
            e.g. a 1 to a 0 or vice versa

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

    def __str__(self):
        return "GENERATION {} : {}".format(self.generation, sorted(self.population, key=self.fitness, reverse=True))

def main():
    ga = GeneticAlgorithm(threshold=15, p=10, r=0.3, m=0.05, length=15)

if __name__ == "__main__":
    main()