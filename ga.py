# coding=utf-8

import multiprocessing
import numpy as np
import random
from operator import attrgetter

global newPopulation

MIN_DISTANCE = 49

class Individual:
# Clase con un contructutor que establece el cromosoma, el fitness (que depende de la similitud entre individuos) y en nominal fitness (real) 
        def __init__ (self, chromosome, fitness=None):
            self.chromosome = chromosome
            self.fitness = fitness
            self.nominal_fitness = 0

        def __lt__ (self, other): # Se comprueba si el fitness del self es menor que el del recibido por parámetro
            return self.fitness < other.fitness

        def __iter__ (self): # Devuelve una versión iterable del cromosoma del individuo
            return iter(self.chromosome)

        def __str__ (self): # Devuelve un string que contiene el fitness y el cromosoma del individuo
            return str('{} (adj. {}) <- {}').format(self.nominal_fitness, self.fitness, ''.join([str(g) for g in self.chromosome]))

        def __repr__ (self): # Imprime el individuo. TODO: Cuál es la diferencia con __str__()?
            return self.__str__()

        def __len__ (self): # len(chromosome)
            return len(self.chromosome)

        def __getitem__ (self, item): # Devuelve un cierto valor del cromosoma al indicarle la posición
            return self.chromosome[item]

class GeneticAlgorithm:
# Clase con un constructor que establece una función, un alfabeto, tamaño de la población (100), longitud del cromosoma(64), tamaño de torne(4), ratio de mutación (0.01)  individuos en la élite(1), sentido del progreso(maximizar o minimizar), tipo de reproducción(un punto), argumentos del cruce, numero de hilos(1) e inicialización(true)
    def __init__ (self, function, alphabet=[0,1], population_size=100, chromosome_length=64,
                  tournament_size=4, mutation_rate=0.01, elitist_individuals=1, maximize=True,
                  crossoverOperator='singlePointCrossover', crossoverArgs={}, threads=1, initialize=True):
        self.function = function
        self.alphabet = alphabet
        self.population_size = population_size
        self.chromosome_length = chromosome_length
        self.tournament_size = tournament_size
        self.mutation_rate = mutation_rate
        self.elitist_individuals = elitist_individuals
        self.maximize = maximize
        self.crossoverOperator = crossoverOperator
        self.crossoverArgs = crossoverArgs
        self.threads = multiprocessing.Pool(threads)
        if initialize:
            self.initialize()

    def __getstate__(self):
        self_dict = self.__dict__.copy()
        del self_dict['threads']
        return self_dict

    def initializeIndividual (self): # Crea el cromosoma del individuo de forma aleatoria
        if type(self.chromosome_length) == int:
            return Individual([random.choice(self.alphabet) for g in range(self.chromosome_length)])
        elif type(self.chromosome_length) == tuple:
            return Individual([random.choice(self.alphabet) for g in range(random.randint(*self.chromosome_length))])
        elif type(self.chromosome_length) == list:
            return Individual([random.choice(self.alphabet) for g in range(random.choice(self.chromosome_length))])

    def initializePopulation(self): # Crea population_size individuos de manera aleatoria
        population = []
        i = 0
        while i < self.population_size:
            individual = self.initializeIndividual()
            if(population):
                distance = self.hamming_distance(individual, population)
                if(distance > MIN_DISTANCE):
                    print("Añadido otro individuo " + str(i))
                    population.append(individual)
                    i += 1
            else:
                print("Añadido el primer individuo a la población")
                population.append(individual)
                i += 1
        return population
    
    def hamming_distance(self, individual, population):
        min_hamming_distance = len(individual)
        for individual_b in population:
            ham_distance = self.compare_individuals(individual, individual_b)
            if(ham_distance < min_hamming_distance):
                min_hamming_distance = ham_distance
        return min_hamming_distance
    
    def compare_individuals(self, individual_a, individual_b):
        distance = 0
        for i in range(len(individual_a)):
            if individual_a[i] != individual_b[i]:
                distance += 1
        return distance

    def evaluatePopulation (self, population): # Evalua mediante hilos, con la funcion, el conjunto de cromosomas de la población. En cada individuo de la poblacion se guarda su fitness
        fitnesses = self.threads.map(self.function, map(lambda i : i.chromosome, population))
        for i in range(len(population)):
            population[i].fitness = fitnesses[i]
        return population

    def tournamentSelection (self): # Elije tournament_size individuos de la poblacion. De entre estos individuos elije al mejor y lo devuelve
        return sorted(random.sample(self.population, self.tournament_size), reverse=self.maximize)[0] #TODO: Elije en función del ponderado o del nominal??

    def elitism (self): # Elije los elitist_individuals de la población y los devuelve como elite de la misma
        return sorted(self.population, key=attrgetter('nominal_fitness'), reverse=self.maximize)[:self.elitist_individuals]

    def singlePointCrossover (self, individual1, individual2): # Elige un punto al azar y devuelve los dos individuos fruto del cruce
        point = random.randint(0, min(len(individual1), len(individual2)))
        return (Individual(individual1[:point] + individual2[point:]), Individual(individual2[:point] + individual1[point:]))

    def variableMultiPointCrossover (self, individual1, individual2, minPoints, maxPoints): # Devuelve los dos individuos fruto del cruce de varios puntos
        points = sorted(np.random.choice(range(0, min(len(individual1), len(individual2))), size=random.randint(minPoints, maxPoints), replace=False))
        inds = (individual1, individual2)
        newInd1, newInd2 = [], []
        for p in points:
            newInd1 +=  inds[0][len(newInd1):p]
            newInd2 +=  inds[1][len(newInd2):p]
            inds = inds[::-1]
        newInd1 += inds[0][points[-1]:]
        newInd2 += inds[1][points[-1]:]
        return (Individual(newInd1), Individual(newInd2))

    def flipMutation (self, individual): # Mutacion aleatoria sobre algunos genes
        return Individual([g if random.random() >= self.mutation_rate else random.choice(list(set(self.alphabet) - set([g]))) for g in individual])

    def initialize (self): # Inicializa la población y la evalua
        self.population = self.initializePopulation()
        self.population = self.evaluatePopulation(self.population)

    def runGeneration (self): # Ejecuta una generacion. Elije un elite, selecciona n individuos, realiza el cruce para cada dos individuos de la seleccion y realiza la mutacion.
        # Despues evalua la poblacion. Por ultimo crea la poblacion final al mezclar la elite y la poblacion mutada. Devuelve el mejor individuo de esta poblacion.
        global newPopulation
        elite = self.elitism()
        print(self.population)
        newPopulation = [self.tournamentSelection() for i in range(self.population_size)]
        newPopulation = list(sum([eval('self.{}(newPopulation[i], newPopulation[i+1], **self.crossoverArgs)'.format(self.crossoverOperator)) for i in range(0, self.population_size, 2)], ()))
        newPopulation = [self.flipMutation(individual) for individual in newPopulation]
        newPopulation = self.evaluatePopulation(newPopulation)
        self.population = elite + newPopulation[self.elitist_individuals:]
        return self.elitism()[0]
