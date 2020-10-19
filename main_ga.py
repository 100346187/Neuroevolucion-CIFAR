
# coding: utf-8

# In[1]:


import getopt
import multiprocessing
import subprocess
import sys
from operator import attrgetter

global EXPERIMENT
EXPERIMENT = "a_p"
global EXPERIMENT_NO
EXPERIMENT_NO = 3


# In[2]:


def gray2bin (bits): # Convierte un numero en gray a uno en binario
    b = [bits[0]]
    for nextb in bits[1:]:
        b.append(b[-1] ^ nextb)
    return b


# In[3]:


def bin2int (bits): # Convierte un numero en binario a entero
    return int(''.join([str(b) for b in bits]), 2)


# In[4]:

"""
def decode (chr):
    props = {}
    ''' Input parameters '''
    props['batch']  = [25, 50, 100, 200][bin2int(gray2bin(chr[0:2]))]

    ''' Convolutional parameters '''
    props['nconv']  = 1  +  bin2int(gray2bin(chr[2:4]))       # [1-4]
    for nc in range(1, props['nconv']+1):
        s = 4 + 10 * (nc-1)
        props['nk{}'.format(nc)] = 2 ** (1 + bin2int(gray2bin(chr[s:s+3])))  # [2, 4, 8, 16, 32, 64, 128, 256]
        props['nr{}'.format(nc)] = 2 + bin2int(gray2bin(chr[s+3:s+6])) # [2-9]
        props['nc{}'.format(nc)] = props['nr{}'.format(nc)]
        props['pr{}'.format(nc)] = 1 + bin2int(gray2bin(chr[s+6:s+9])) # [2-9]
        props['pc{}'.format(nc)] = props['pr{}'.format(nc)]
        props['ca{}'.format(nc)] = ['relu', 'linear'][bin2int(gray2bin(chr[s+9:s+10]))]

    ''' Dense and recurrent parameters '''
    props['ndense'] = 1 +  bin2int(gray2bin(chr[44:45]))     # [1-2]
    for nd in range(1, props['ndense']+1):
        s = 45 + 9 * (nd-1)
        props['dt{}'.format(nd)] = ['rnn', 'lstm', 'gru', 'dense'][bin2int(chr[s:s+2])]
        props['nn{}'.format(nd)] = 2 ** (3 + bin2int(gray2bin(chr[s+2:s+5])))  # [8, 16, 32, 64, 128, 256, 512, 1024]
        props['da{}'.format(nd)] = ['relu', 'linear'][bin2int(gray2bin(chr[s+5:s+6]))]
        props['re{}'.format(nd)] = ['None', 'l1', 'l2', 'l1l2'][bin2int(chr[s+6:s+8])]
        props['dr{}'.format(nd)] = [0, 0.5][bin2int(chr[s+8:s+9])]

    ''' Optimizer '''
    props['opt'] = ['sgd', 'momentum', 'nesterov', 'adagrad', 'adamax', 'adam', 'adadelta', 'rmsprop'][bin2int(chr[63:66])]
    props['lr']  = [1E-5, 5E-5, 1E-4, 5E-4, 1E-3, 5E-3, 1E-2, 5E-2][bin2int(chr[66:69])]

    return props
"""

def decode (chr):
    props = {}
    ''' Input parameters '''
    props['batch']  = [25, 50, 100, 200][bin2int(gray2bin(chr[0:2]))]

    ''' Convolutional parameters '''
    props['nconv']  = 1  +  bin2int(gray2bin(chr[2:5]))       # [1-4]
    for nc in range(1, props['nconv']+1):
        s = 5 + 10 * (nc-1)
        props['nk{}'.format(nc)] = 2 ** (1 + bin2int(gray2bin(chr[s:s+3])))  # [2, 4, 8, 16, 32, 64, 128, 256]
        props['nr{}'.format(nc)] = 2 + bin2int(gray2bin(chr[s+3:s+6])) # [2-9]
        props['nc{}'.format(nc)] = props['nr{}'.format(nc)]
        props['pr{}'.format(nc)] = 1 + bin2int(gray2bin(chr[s+6:s+9])) # [2-9]
        props['pc{}'.format(nc)] = props['pr{}'.format(nc)]
        props['ca{}'.format(nc)] = ['relu', 'linear'][bin2int(gray2bin(chr[s+9:s+10]))]
    
    ''' Dense and recurrent parameters '''
    props['ndense'] = 1 +  bin2int(gray2bin(chr[85:86]))     # [1-2]
    for nd in range(1, props['ndense']+1):
        s = 86 + 9 * (nd-1)
        props['dt{}'.format(nd)] = ['rnn', 'lstm', 'gru', 'dense'][bin2int(chr[s:s+2])]
        props['nn{}'.format(nd)] = 2 ** (3 + bin2int(gray2bin(chr[s+2:s+5])))  # [8, 16, 32, 64, 128, 256, 512, 1024]
        props['da{}'.format(nd)] = ['relu', 'linear'][bin2int(gray2bin(chr[s+5:s+6]))]
        props['re{}'.format(nd)] = ['None', 'l1', 'l2', 'l1l2'][bin2int(chr[s+6:s+8])]
        props['dr{}'.format(nd)] = [0, 0.5][bin2int(chr[s+8:s+9])]

    ''' Optimizer '''
    props['opt'] = ['sgd', 'momentum', 'nesterov', 'adagrad', 'adamax', 'adam', 'adadelta', 'rmsprop'][bin2int(chr[104:107])]
    props['lr']  = [1E-5, 5E-5, 1E-4, 5E-4, 1E-3, 5E-3, 1E-2, 5E-2][bin2int(chr[107:110])]

    return props

# In[5]:


def similarity (ind1, ind2): # Devuelve el numero de bits que coinciden entre los individuos entre la longitud del individuo
    if ind1['nconv'] != ind2['nconv'] or ind1['ndense'] != ind2['ndense']: # Da un numero entre 0 y 1 que indica la similitud
      return 0
    else:
      return float(sum([1 if ind1[k] == ind2[k] else 0 for k in ind1.keys()]) - 2) / (len(ind1) - 2)


# In[6]:


def similarityWithPopulation (ind, pop): # Devuelve la similitud entre el individuo y la población. También entre 0 y 1
    return (sum([similarity(ind, other) for other in pop]) - 1) / (len(pop) - 1)


# In[7]:


def serialize_population(individuals, it): # Escribe los individuos de la población en un archivo indicando el experimento realizado
    import pickle as cp
    global EXPERIMENT, EXPERIMENT_NO
    cp.dump(individuals, open('population_exp{}_it{}.pkl'.format(EXPERIMENT_NO, it), 'wb'), protocol=cp.HIGHEST_PROTOCOL)


# In[8]:


def load_population(file): # Carga a los individuos de la población de un archivo dado
    import pickle as cp
    return cp.load(open(file))


# In[9]:


def fit_mnist (chr): # Evalua al individuo de forma paralela con dos hilos con la funcion th_mnist_val, en un fichero diferente
    global EXPERIMENT
    args = ""
    for k,v in decode(chr).items():
        args += "--{}={} ".format(k, v)
    args += "--device=NONE --sample=0.5 --epochs=5 --experiment={}".format(EXPERIMENT)
    from subprocess import Popen, PIPE
    #print(args)
    p = Popen(['python3', 'tf_mnist_val.py'] + args.split(' '), stdout=PIPE, stderr=PIPE)
    o, e = p.communicate()
    #print(e)
    #print(o)
    fit = float(o)
    #print(fit)
    return fit


# In[10]:

devs = multiprocessing.Queue()
devs.put(0)
devs.put(1)


def fit_cifar (chr): # Evalua al individuo de forma paralela con dos hilos con la funcion th_mnist_val, en un fichero diferente
    global EXPERIMENT, devs
    d = devs.get()
    args = ""
    for k,v in decode(chr).items():
        args += "--{}={} ".format(k, v)
    args += "--device={} --sample=0.5 --epochs=16 --experiment={}".format(d, EXPERIMENT)
    from subprocess import Popen, PIPE
    #print(args)
    p = Popen(['python3', 'tf_cifar_val.py'] + args.split(' '), stdout=PIPE, stderr=PIPE)
    o, e = p.communicate()
    #print(e)
    #print(o)
    fit = float(o)
    #print(fit)
    devs.put(d)
    return fit


# In[ ]:


import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
    
    
import ga 
from timeit import default_timer as timer
print("Inicio de la creación del algoritmo")
GA = ga.GeneticAlgorithm(fit_cifar, threads=2, population_size=50, tournament_size=3, chromosome_length=110, crossoverOperator='variableMultiPointCrossover', crossoverArgs={'minPoints': 3, 'maxPoints': 10}, mutation_rate=0.015, initialize=True)
print("Fin de la creación del algoritmo")
# Fixes the random initialization in case it is zero.
print("Limpieza de individuos defectuosos")
while True: # Bucle infinito hasta que toda la población tenga un fitness distinto de cero
    valid = list(filter(lambda i : i.fitness != 0, GA.population))
    print("Se tienen {} individuos de un total de {}".format(len(valid), GA.population_size))
    if len(valid) == GA.population_size:
        break
    zeros = [GA.initializeIndividual() for i in range(GA.population_size-len(valid))]
    zeros = GA.evaluatePopulation(zeros)
    GA.population = valid + zeros

print("Inicio del algoritmo propiamente dicho")
    
MAX_GENERATIONS = 100   # Numero de generaciones
MAX_GENERATIONS_WITHOUT_IMPROVEMENTS = 30   # Numero maximo de generaciones sin mejora

iterationsWithoutImprovements = 0
bestFitness = float('-inf')

for i in range(MAX_GENERATIONS):
    print("Iteration {}".format(i))
    end = timer()

    # Performs niching
    for ind in GA.population:
        ind.nominal_fitness = ind.fitness
        ind.fitness *= (1 - similarityWithPopulation(decode(ind.chromosome), [decode(o.chromosome) for o in GA.population])) #TODO: Recordar niching

    GA.population.sort(key=attrgetter('nominal_fitness'), reverse=GA.maximize)
    if GA.population[0].nominal_fitness >= bestFitness: # Se comprueba si ha mejorado el fitness del mejor
        bestFitness = GA.population[0].nominal_fitness
        iterationsWithoutImprovements = 0
    else:
        iterationsWithoutImprovements += 1

    serialize_population(GA.population, i) # Se imprime la iteracion en fichero

    if 'start' in locals(): # Se imprime el tiempo
        print("  Took %f s." % (end - start))

    # Checks for the stop condition
    print("  {} iterations without improvements.".format(iterationsWithoutImprovements))
    if iterationsWithoutImprovements >= MAX_GENERATIONS_WITHOUT_IMPROVEMENTS:
        break

    start = timer()
    GA.runGeneration() # Se pasa una generacion mas

