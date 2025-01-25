###############################################################################
#                               LIBRARIES                                     #
###############################################################################
from pprint import pprint
from datetime import datetime
from itertools import combinations

import numpy as np
import pandas as pd
import math
from scipy import constants
from mendeleev import element
from deap import tools
import itertools


from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.ensemble import GradientBoostingRegressor

from glas import GlassSearcher as Searcher
from glas.constraint import Constraint
from glas.predict import Predict

import warnings
warnings.filterwarnings("ignore")

###############################################################################
#                               PROPERTIES                                    #
###############################################################################
descriptors=pd.read_csv('Elemental_properties.csv')
dfHmix = pd.read_excel(r"Hmix.xlsx", index_col=0)

elements=np.array(descriptors['element'])
Electronegativity=np.array(descriptors['Electronegativity'])
VEC=np.array(descriptors['NValence'])
AtomicRadius=np.array(descriptors['AtomicRadius'])
LatticeConstant=np.array(descriptors['LatticeConstant'])
MeltT=np.array(descriptors['MeltT'])
Meq=np.array(descriptors['Meq'])
Bo=np.array(descriptors['Bo'])
Md=np.array(descriptors['Md'])

###############################################################################
#                               FUNCTIONS                                     #
###############################################################################
def normalizer (alloys):
    total=np.sum(alloys,axis=1).reshape((-1,1))
    norm_alloys=alloys/total
    return norm_alloys
###############################################################################
#                               MASS CONVERSION                               #
###############################################################################

# Obter massas atômicas dos elementos utilizando a biblioteca mendeleev
massas_atomicas = np.array([element(el).atomic_weight for el in elements])

def atomic_to_mass_percent(alloys):
    """
    Converte as frações atômicas em frações mássicas.
    """
    # Multiplica cada fração atômica pela respectiva massa atômica
    mass_fractions = alloys*massas_atomicas
    
    # Soma total das massas
    total_mass = np.sum(mass_fractions, axis=1).reshape((-1, 1))
    
    # Calcula porcentagem mássica
    mass_percent = (mass_fractions / total_mass)*100
    
    return mass_percent
###############################################################################
#                               DESCRIPTORS                                   #
###############################################################################
def Meq_avg (alloys):
    Meq_avg = (Meq*alloys).sum(axis=1)
    return Meq_avg

def Meq_std_dev (alloys):
    Meq_avg = (Meq*alloys).sum(axis=1)
    variance = ((alloys * (Meq - Meq_avg[:, None])**2).sum(axis=1))
    Meq_std_dev = np.sqrt(variance)
    return Meq_std_dev

def Bo_avg (alloys):
    Bo_avg = (Bo*alloys).sum(axis=1)
    return Bo_avg

def Bo_std_dev (alloys):
    Bo_avg = (Bo*alloys).sum(axis=1)
    variance = ((alloys * (Bo - Bo_avg[:, None])**2).sum(axis=1))
    Bo_std_dev = np.sqrt(variance)
    return Bo_std_dev

def Md_avg (alloys):
    Md_avg = (Md*alloys).sum(axis=1)
    return Md_avg

def Md_std_dev (alloys):
    Md_avg = (Md*alloys).sum(axis=1)
    variance = ((alloys * (Md - Md_avg[:, None])**2).sum(axis=1))
    Md_std_dev = np.sqrt(variance)
    return Md_std_dev

def Lattice_Constant_avg (alloys):
    Lattice_Constant_avg = (LatticeConstant*alloys).sum(axis=1)
    return Lattice_Constant_avg

def Lattice_Constant_std_dev (alloys):
    Lattice_Constant_avg = (LatticeConstant*alloys).sum(axis=1)
    variance = ((alloys * (LatticeConstant - Lattice_Constant_avg[:, None])**2).sum(axis=1))
    Lattice_Constant_std_dev = np.sqrt(variance)
    return Lattice_Constant_std_dev

def atomic_radius_avg (alloys):
    atomic_radius_avg = (AtomicRadius*alloys).sum(axis=1)
    return atomic_radius_avg

def atomic_radius_std_dev (alloys):
    atomic_radius_avg = (AtomicRadius*alloys).sum(axis=1)
    variance = ((alloys * (AtomicRadius - atomic_radius_avg[:, None])**2).sum(axis=1))
    atomic_radius_std_dev = np.sqrt(variance)
    return atomic_radius_std_dev


def melting_temperature_avg (alloys):
    melting_temperature_avg = (MeltT*alloys).sum(axis=1)   
    return melting_temperature_avg

def melting_temperature_std_dev (alloys):
    melting_temperature_avg = (MeltT*alloys).sum(axis=1)
    variance = ((alloys * (MeltT - melting_temperature_avg[:, None])**2).sum(axis=1))
    melting_temperature_std_dev = np.sqrt(variance)
    return melting_temperature_std_dev

def VEC_avg (alloys):
    VEC_avg = (VEC*alloys).sum(axis=1)   
    return VEC_avg

def VEC_std_dev (alloys):
    VEC_avg = (VEC*alloys).sum(axis=1)
    variance = ((alloys * (VEC - VEC_avg[:, None])**2).sum(axis=1))
    VEC_std_dev = np.sqrt(variance)
    return VEC_std_dev

def Electronegativity_avg (alloys):
    Electronegativity_avg = (Electronegativity*alloys).sum(axis=1)   
    return Electronegativity_avg

def Electronegativity_std_dev (alloys):
    Electronegativity_avg = (Electronegativity*alloys).sum(axis=1)
    variance = ((alloys * (Electronegativity - Electronegativity_avg[:, None])**2).sum(axis=1))
    Electronegativity_std_dev = np.sqrt(variance)
    return Electronegativity_std_dev

def parSid (alloys):
    alloys = np.array(alloys)
    with np.errstate(divide='ignore', invalid='ignore'):
        Sid = -(alloys * np.log(alloys)).sum(axis=1)
        Sid = np.nan_to_num(Sid)  
    return Sid

#Phi
arrAtomicSize = np.array(descriptors['AtomicRadius']) * 100
arrMeltingT = np.array(descriptors['MeltT'])
elements = np.array(descriptors['element'])

def normalizer(alloys):
    total = np.sum(alloys, axis=1).reshape((-1, 1))
    norm_alloys = alloys / total
    return norm_alloys

def Smix(compNorm):
    x = np.sum(np.nan_to_num((compNorm) * np.log(compNorm)), axis=1)
    Smix = -constants.R * 10 ** -3 * x
    return Smix

def Tm(compNorm):
    Tm = np.sum(compNorm * arrMeltingT, axis=1)
    return Tm

def Hmix(compNorm):
    elements_present = compNorm.sum(axis=0).astype(bool)
    compNorm = compNorm[:, elements_present]
    element_names = elements[elements_present]
    Hmix = np.zeros(compNorm.shape[0])
    for i, j in combinations(range(len(element_names)), 2):
        Hmix = (
                Hmix
                + 4
                * dfHmix[element_names[i]][element_names[j]]
                * compNorm[:, i]
                * compNorm[:, j]
        )
    return Hmix

def Sh(compNorm):
    Sh = abs(Hmix(compNorm)) / Tm(compNorm)
    return Sh

def csi_i(compNorm, AP):
    supportValue = np.sum((1 / 6) * math.pi * (arrAtomicSize * 2) ** 3 * compNorm, axis=1)
    rho = AP / supportValue
    csi_i = (1 / 6) * math.pi * rho[:, None] * (arrAtomicSize * 2) ** 3 * compNorm
    return csi_i

def deltaij(i, j, newCompNorm, newArrAtomicSize, csi_i_newCompNorm, AP):
    element1Size = newArrAtomicSize[i] * 2
    element2Size = newArrAtomicSize[j] * 2
    deltaij = ((csi_i_newCompNorm[:, i] * csi_i_newCompNorm[:, j]) ** (1 / 2) / AP) * (
                ((element1Size - element2Size) ** 2) / (element1Size * element2Size)) * (
                          newCompNorm[:, i] * newCompNorm[:, j]) ** (1 / 2)
    return deltaij

def y1_y2(compNorm, AP):
    csi_i_compNorm = csi_i(compNorm, AP)
    elements_present = compNorm.sum(axis=0).astype(bool)
    newCompNorm = compNorm[:, elements_present]
    newCsi_i_compNorm = csi_i_compNorm[:, elements_present]
    newArrAtomicSize = arrAtomicSize[elements_present]
    y1 = np.zeros(newCompNorm.shape[0])
    y2 = np.zeros(newCompNorm.shape[0])
    for i, j in combinations(range(len(newCompNorm[0])), 2):
        deltaijValue = deltaij(i, j, newCompNorm, newArrAtomicSize, newCsi_i_compNorm, AP)
        y1 += deltaijValue * (newArrAtomicSize[i] * 2 + newArrAtomicSize[j] * 2) * (
                    newArrAtomicSize[i] * 2 * newArrAtomicSize[j] * 2) ** (-1 / 2)
        y2_ = np.sum((newCsi_i_compNorm / AP) * (
                    ((newArrAtomicSize[i] * 2 * newArrAtomicSize[j] * 2) ** (1 / 2)) / (newArrAtomicSize * 2)), axis=1)
        y2 += deltaijValue * y2_
    return y1, y2

def y3(compNorm, AP):
    csi_i_compNorm = csi_i(compNorm, AP)
    x = (csi_i_compNorm / AP) ** (2 / 3) * compNorm ** (1 / 3)
    y3 = (np.sum(x, axis=1)) ** 3
    return y3

def Z(compNorm, AP):
    y1Values, y2Values = y1_y2(compNorm, AP)
    y3Values = y3(compNorm, AP)
    Z = ((1 + AP + AP ** 2) - 3 * AP * (y1Values + y2Values * AP) - AP ** 3 * y3Values) * (1 - AP) ** (-3)
    return Z

def eq4B(compNorm, AP):
    y1Values, y2Values = y1_y2(compNorm, AP)
    y3Values = y3(compNorm, AP)
    eq4B = -(3 / 2) * (1 - y1Values + y2Values + y3Values) + (3 * y2Values + 2 * y3Values) * (1 - AP) ** -1 + (
                3 / 2) * (1 - y1Values - y2Values - (1 / 3) * y3Values) * (1 - AP) ** -2 + (y3Values - 1) * np.log(
        1 - AP)
    return eq4B

def Se(compNorm, AP):
    Se = (eq4B(compNorm, AP) - np.log(Z(compNorm, AP)) - (3 - 2 * AP) * (1 - AP) ** -2 + 3 + np.log(
        (1 + AP + AP ** 2 - AP ** 3) * (1 - AP) ** -3)) * constants.R * 10 ** -3
    return Se

def parPhi(alloys):
    compNorm = normalizer(alloys)
    SeBCC = Se(compNorm, 0.68)
    SeFCC = Se(compNorm, 0.74)
    SeMean = (abs(SeBCC) + abs(SeFCC)) / 2
    phi = (Smix(compNorm) - Sh(compNorm)) / SeMean
    return phi


###############################################################################
#                           MACHINE LEARNING                                  #
###############################################################################

#GBR E
E = pd.read_csv('E_ML_descriptors.csv')
E_descriptors = ['Sid','Electronegativity_avgs_w','Electronegativity_std_devs_w',
                 'VEC_avgs_w','VEC_std_devs_w','AtomicRadius_avgs_w','AtomicRadius_std_devs_w',
                 'LatticeConstant_avgs_w','LatticeConstant_std_devs_w','MeltT_avgs_w',
                 'MeltT_std_devs_w','phi','Meq_avgs_w','Meq_std_devs_w',
                 'Md_avgs_w','Md_std_devs_w','Bo_avgs_w','Bo_std_devs_w']
scaler = StandardScaler()
x_E = E[E_descriptors]
x_E = scaler.fit_transform(x_E)
y_E = E['E']
gbr = GradientBoostingRegressor(learning_rate= 0.1,
 max_depth= 3,
 min_samples_leaf= 1,
 min_samples_split= 2,
 n_estimators= 100)
gbr.fit(x_E,np.ravel(y_E))
             
def parE (alloys):
    Sid = parSid(alloys)
    Electronegativity_avgs_w = Electronegativity_avg(alloys)
    Electronegativity_std_devs_w = Electronegativity_std_dev(alloys)
    VEC_avgs_w = VEC_avg(alloys)
    VEC_std_devs_w = VEC_std_dev(alloys)
    AtomicRadius_avgs_w = atomic_radius_avg(alloys)
    AtomicRadius_std_devs_w = atomic_radius_std_dev(alloys)
    LatticeConstant_avgs_w = Lattice_Constant_avg(alloys)
    LatticeConstant_std_devs_w = Lattice_Constant_std_dev(alloys)
    MeltT_avgs_w = melting_temperature_avg(alloys)
    MeltT_std_devs_w = melting_temperature_std_dev(alloys)
    phi = parPhi(alloys)
    Meq_avgs_w = Meq_avg(alloys)
    Meq_std_devs_w = Meq_std_dev(alloys)
    Md_avgs_w = Md_avg(alloys)
    Md_std_devs_w = Md_std_dev(alloys)
    Bo_avgs_w = Bo_avg(alloys)
    Bo_std_devs_w = Bo_std_dev(alloys)
    
    alloys_features = np.column_stack([Sid,Electronegativity_avgs_w,Electronegativity_std_devs_w,VEC_avgs_w,VEC_std_devs_w,AtomicRadius_avgs_w,
                                       AtomicRadius_std_devs_w,LatticeConstant_avgs_w,LatticeConstant_std_devs_w,MeltT_avgs_w,MeltT_std_devs_w, 
                                       phi,Meq_avgs_w,Meq_std_devs_w,Md_avgs_w, Md_std_devs_w,Bo_avgs_w,Bo_std_devs_w])   
    alloys_scale = scaler.transform(alloys_features)
    E_prediction = gbr.predict(alloys_scale)
    return E_prediction

###############################################################################
#                             Predict Class                                   #
###############################################################################

class PredictE(Predict):
    def __init__(self, all_elements, **kwargs):
        super().__init__()
        self.domain = {el: [0,1] for el in all_elements}

    def predict(self, population_dict):
        alloys=population_dict['population_array']
        alloys=normalizer(alloys)
        value=parE(alloys)
        return value 

    def get_domain(self):
        return self.domain

    def is_within_domain(self, population_dict):
        return np.ones(len(population_dict['population_array'])).astype(bool)

###############################################################################
#                           Constraint Class                                  #
###############################################################################

class ConstraintElements(Constraint):
    def __init__(self, config, compound_list, **kwargs):
        super().__init__()
        self.config = config
        elemental_domain = {el: [0, 1] for el in compound_list}
        for el in config:
            elemental_domain[el] = config[el]
        self.elemental_domain = elemental_domain

    def compute(self, population_dict, base_penalty):
        norm_pop = normalizer(population_dict['population_array'])
        distance = np.zeros(population_dict['population_array'].shape[0])

        for n, el in enumerate(self.elemental_domain):
            el_atomic_frac = norm_pop[:, n]
            el_domain = self.elemental_domain.get(el, [0, 0])

            logic1 = el_atomic_frac > el_domain[1]
            distance[logic1] += el_atomic_frac[logic1] - el_domain[1]

            logic2 = el_atomic_frac < el_domain[0]
            distance[logic2] += el_domain[0] - el_atomic_frac[logic2]

        logic = distance > 0
        distance[logic] = (100 * distance[logic])**2 + base_penalty
        penalty = distance

        return penalty

class ConstraintMeq(Constraint):
    def __init__(self, config, **kwargs):
        super().__init__()
        self.config = config

    def compute(self, population_dict, base_penalty):
        alloys=population_dict['population_array']
        alloys=normalizer(alloys)
        value = Meq_avg(alloys)
        bad = value < self.config['min']

        distance_min = self.config['min'] - value
        distance = np.zeros(len(value))
        distance[bad] += distance_min[bad]

        penalty = bad * base_penalty + distance**2
        return penalty
###############################################################################
#                            Search Class                                     #
###############################################################################

class Searcher(Searcher):
    def __init__(self, config, design, constraints={}, run_num=0):

        self.run_num = run_num
        super().__init__(config, design, constraints)

    def callback(self):
        best_fitness = min([ind.fitness.values[0] for ind in self.population])
        print(
            'Finished generation {1}/{0}. '.format(
                str(self.generation).zfill(3),
                str(self.run_num).zfill(2)
            ),
            f'Best fitness is {best_fitness:.3g}. '
        )

        if self.generation % self.report_frequency == 0:
            if best_fitness < self.base_penalty:
                best_ind = tools.selBest(self.population, 1)[0]
                print('\nBest individual in this population (in weigth%):')
                self.report_dict(best_ind, verbose=True)

###############################################################################
#                         Design, Constraint & Config                         #
###############################################################################

design = {
    'E': {
        'class': PredictE,
        'name': 'E',
        'use_for_optimization': True,
        'config': {
            'min': 0,
            'max': 200,
            'objective': 'minimize',
            'weight': 1,
        }
    }
}
constraints = {
    
    'elements': {
        'class': ConstraintElements,
        'config': {
            'Cr': [0.00, 0.10],
            'Cu': [0.00, 0.10],
            'Fe': [0.00, 0.10],
            'Mn': [0.00, 0.10],
            'Mo': [0.00, 0.10],
            'Nb': [0.00, 0.30],
            'Si': [0.00, 0.00], 
            'Sn': [0.00, 0.10],
            'Ta': [0.00, 0.10],
            'Ti': [0.50, 1.0],
            'V': [0.00, 0.10],
            'Zr': [0.00, 0.30],
        },
    },
    'Meq': {
        'class': ConstraintMeq,
        'config': {
            'min': 0.1,
        },
    },
}
config = {
    'num_generations':1000,
    'population_size': 100,
    'hall_of_fame_size': 1,
    'num_repetitions': 100,
    'compound_list': list(elements),
}
###############################################################################
#                                    Search                                   #
###############################################################################
all_hof = []
for i in range(config['num_repetitions']):
    S = Searcher(config, design, constraints, i+1)
    S.start()
    S.run(config['num_generations'])
    all_hof.append(S.hof)
    fim = datetime.now()

###############################################################################
#                                 Print Report                                #
###############################################################################

print()
print('--------  REPORT -------------------')
print()
print('--------  Design Configuration -------------------')
pprint(config)
print()
pprint(design)
print()
print('--------  Constraints -------------------')
pprint(constraints)
print()

for p, hof in enumerate(all_hof):
    print()
    print(f'------- RUN {p+1} -------------')
    print()
    for n, ind in enumerate(hof):
        print(f'Position {n+1} (mol%)')
        print(f'Fitness: {S.fitness_function([ind])[0]:5f}')
        S.report_dict(ind, verbose=True)

###############################################################################
#                                 File Report                                 #
##############################################################################+
report_data = []
dict_functions = {
    'E': parE,
    'Meq': Meq_avg,
}
for population_index, hall_of_fame in enumerate(all_hof):
    normalized_alloys = normalizer(hall_of_fame)
    df_alloys = pd.DataFrame(normalized_alloys * 100, columns=list(elements))

    fitness_values = S.fitness_function(normalized_alloys)
    df_fitness = pd.DataFrame(fitness_values, columns=["fitness"])
    df_alloys = pd.concat([df_alloys, df_fitness], axis=1)

    for property_name, property_function in dict_functions.items():
        property_values = property_function(normalized_alloys)
        df_properties = pd.DataFrame(property_values, columns=[property_name])
        df_alloys = pd.concat([df_alloys, df_properties], axis=1)
    
    report_data.append(df_alloys)

report_df = pd.concat(report_data, ignore_index=True)
report_df.to_csv('GA_results.csv', index=False)