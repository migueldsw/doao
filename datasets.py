import numpy as np
import math
import matplotlib.pyplot as plt
from numpy import *
from sklearn.datasets import fetch_mldata
from sklearn import datasets
from sklearn import preprocessing
import urllib

def class2int_iris(s):
    
    if s == b'Iris-setosa':
        return 0
    elif s == b'Iris-versicolor':
        return 1
    else:
        return 2

def class2int_balance(s):
    
    if s == b'L':
        return 0
    elif s == b'B':
        return 1
    else:
        return 2

def class2int_vehicle(s):
    
    if s == b'opel':
        return 0
    elif s == b'saab':
        return 1
    elif s == b'bus':
        return 2
    else:
        return 3

        
def class2int_ecoli(s):
    
    if s == b'cp':
        return 0
    elif s == b'im':
        return 1
    elif s == b'imS':
        return 2
    elif s == b'imL':
        return 3
    elif s == b'imU':
        return 4
    elif s == b'om':
        return 5
    elif s == b'omL':
        return 6
    else:
        return 7

def class2int_yeast(s):
    
    if s == b'CYT':
        return 0
    elif s == b'NUC':
        return 1
    elif s == b'MIT':
        return 2
    elif s == b'ME3':
        return 3
    elif s == b'ME2':
        return 4
    elif s == b'ME1':
        return 5
    elif s == b'EXC':
        return 6
    elif s == b'VAC':
        return 7
    elif s == b'POX':
        return 8
    else:
        return 9

def class2int_segment(s):
    
    if s == b'BRICKFACE':
        return 0
    elif s == b'SKY':
        return 1
    elif s == b'FOLIAGE':
        return 2
    elif s == b'CEMENT':
        return 3
    elif s == b'WINDOW':
        return 4
    elif s == b'PATH':
        return 5
    else:
        return 6

def class2int_landcover(s):
    
    if s == b'tree ':
        return 0
    elif s == b'grass ':
        return 1
    elif s == b'soil ':
        return 2
    elif s == b'concrete ':
        return 3
    elif s == b'asphalt ':
        return 4
    elif s == b'building ':
        return 5
    elif s == b'car ':
        return 6
    elif s == b'pool ':
        return 7
    else:
        return 8

    
def  data_target(data):
    x = data[:,0:data[0,:].size - 1]
    y = data[:,data[0,:].size - 1]
    return (x,y)

def normalize_columns(data):
    rows, cols = data.shape
    for col in range(0,cols):
        minimo = data[:,col].min()
        maximo = data[:,col].max()
        
        if(minimo != maximo):
            denominador = maximo - minimo
            normazu = (data[:,col] - minimo) / denominador
            data[:,col] = (normazu*2) - 1;
        else:
            data[:,col] = 0
     
#URL's
url_zoo = "http://archive.ics.uci.edu/ml/machine-learning-databases/zoo/zoo.data"
url_wine = "http://archive.ics.uci.edu/ml/machine-learning-databases/wine/wine.data"
url_iris = "http://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"
url_seed = "http://archive.ics.uci.edu/ml/machine-learning-databases/00236/seeds_dataset.txt"
url_glass = "http://archive.ics.uci.edu/ml/machine-learning-databases/glass/glass.data"
url_ecoli = "http://archive.ics.uci.edu/ml/machine-learning-databases/ecoli/ecoli.data"
url_movement = "http://archive.ics.uci.edu/ml/machine-learning-databases/libras/movement_libras.data"
url_balance = "http://archive.ics.uci.edu/ml/machine-learning-databases/balance-scale/balance-scale.data"
url_annealing = "http://archive.ics.uci.edu/ml/machine-learning-databases/annealing/anneal.data"

#raw_data = urllib.request.urlopen(url_seed)

#dataset = np.loadtxt(raw_data, delimiter=",")
#dataset = np.loadtxt('iris.data',delimiter= ',',converters={4: lambda s: class2int(s)},skiprows=1)

# separate the data from the target attributes
#X = dataset[:,0:4]
#y = dataset[:,8]

#iris = datasets.load_iris()
#x, y = iris.data, iris.target

##  READ ZOO  ##
zoo = np.genfromtxt('./Data/zoo.data', delimiter=',', usecols = range(1,18))

##  READ IRIS  ##
iris = np.loadtxt('./Data/iris.data', delimiter=',', converters={4: lambda s: class2int_iris(s)},skiprows=0)

##  READ WINE  ##
wine = np.genfromtxt('./Data/wine.data', delimiter=',')

##  READ SEEED  ##
seed = np.genfromtxt('./Data/seeds_dataset.txt')

##  READ GLASS  ##
glass = np.genfromtxt('./Data/glass.data', delimiter=',', usecols = range(1,11))

##  READ ECOLI  ##
ecoli = np.loadtxt('./Data/ecoli.data', usecols = range(1,9), converters={8: lambda s: class2int_ecoli(s)},skiprows=0)

##  READ BALANCE  ##
balance = np.loadtxt('./Data/balance-scale.data', delimiter=',', converters={0: lambda s: class2int_balance(s)},skiprows=0)

##  READ VOWEL  ##
vowel = np.genfromtxt('./Data/vowel-context.data', usecols = range(3,14))

##  READ YEAST  ##
yeast = np.loadtxt('./Data/yeast.data', usecols = range(1,10), converters={9: lambda s: class2int_yeast(s)},skiprows=0)

##  READ SEGMENT  ##
segment = np.loadtxt('./Data/segmentation.data', delimiter=',', converters={0: lambda s: class2int_segment(s)},skiprows=5)

##  READ MOVIMENT  ##
moviment = np.loadtxt('./Data/movement_libras.data', delimiter=',')

##  READ VEHICLE  ##
vehicle1 = np.loadtxt('./Data/vehicle/xaa.dat', converters={18: lambda s: class2int_vehicle(s)},skiprows=0)
vehicle2 = np.loadtxt('./Data/vehicle/xab.dat', converters={18: lambda s: class2int_vehicle(s)},skiprows=0)
vehicle3 = np.loadtxt('./Data/vehicle/xac.dat', converters={18: lambda s: class2int_vehicle(s)},skiprows=0)
vehicle4 = np.loadtxt('./Data/vehicle/xad.dat', converters={18: lambda s: class2int_vehicle(s)},skiprows=0)
vehicle5 = np.loadtxt('./Data/vehicle/xae.dat', converters={18: lambda s: class2int_vehicle(s)},skiprows=0)
vehicle6 = np.loadtxt('./Data/vehicle/xaf.dat', converters={18: lambda s: class2int_vehicle(s)},skiprows=0)
vehicle7 = np.loadtxt('./Data/vehicle/xag.dat', converters={18: lambda s: class2int_vehicle(s)},skiprows=0)
vehicle8 = np.loadtxt('./Data/vehicle/xah.dat', converters={18: lambda s: class2int_vehicle(s)},skiprows=0)
vehicle9 = np.loadtxt('./Data/vehicle/xai.dat', converters={18: lambda s: class2int_vehicle(s)},skiprows=0)

vehicle = np.concatenate((vehicle1,vehicle2,vehicle3,vehicle4,vehicle5,vehicle6,vehicle7,vehicle8,vehicle9))

##  READ LAND COVER  ##
landcover1 = np.loadtxt('./Data/training.csv', delimiter=',', converters={0: lambda s: class2int_landcover(s)},skiprows=1)
landcover2 = np.loadtxt('./Data/testing.csv', delimiter=',', converters={0: lambda s: class2int_landcover(s)},skiprows=1)

landcover = np.concatenate((landcover1,landcover2))

##  SEPARA EM DATA E TARGET  ##
d_iris, t_iris = data_target(iris)
d_zoo, t_zoo = data_target(zoo)
d_wine, t_wine = wine[:,1:14], wine[:,0]
d_seed, t_seed = data_target(seed)
d_yeast, t_yeast = data_target(yeast)
d_glass, t_glass = data_target(glass)
d_ecoli, t_ecoli = data_target(ecoli)
d_balance, t_balance = balance[:,1:5], balance[:,0]
d_vowel, t_vowel = data_target(vowel)
d_segment, t_segment = segment[:,1:20], segment[:,0]
d_moviment, t_moviment = data_target(moviment)
d_vehicle, t_vehicle = data_target(vehicle)
d_landcover, t_landcover = landcover[:,1:148], landcover[:,0]

normalize_columns(d_iris)
normalize_columns(d_zoo)
normalize_columns(d_wine)
normalize_columns(d_seed)
normalize_columns(d_yeast)
normalize_columns(d_glass)
normalize_columns(d_ecoli)
normalize_columns(d_balance)
normalize_columns(d_vowel)
normalize_columns(d_segment)
normalize_columns(d_moviment)
normalize_columns(d_vehicle)
normalize_columns(d_landcover)


DATA = {
    'iris': (d_iris, t_iris),
    'zoo': (d_zoo, t_zoo),
    'wine': (d_wine, t_wine),
    'seed': (d_seed, t_seed),
    'yeast': (d_yeast, t_yeast),
    'glass': (d_glass, t_glass),
    'ecoli': (d_ecoli, t_ecoli),
    'balance': (d_balance, t_balance),
    'vowel': (d_vowel, t_vowel),
    'segment': (d_segment, t_segment),
    'moviment': (d_moviment, t_moviment),
    'vehicle': (d_vehicle, t_vehicle),
    'landcover': (d_landcover, t_landcover),
    }
