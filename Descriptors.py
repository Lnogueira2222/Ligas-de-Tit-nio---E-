#Libraries
import pandas as pd
import numpy as np
import itertools
from itertools import combinations
from mendeleev import element
from scipy import constants
import math
import xlsxwriter

#CSV files
database = pd.read_csv('Descriptw.csv')
compositions = database.iloc[:, :-1]
properties = pd.read_csv('Elemental_properties.csv')
dfHmix = pd.read_excel(r"Hmix.xlsx", index_col=0)

#Composition normalizer
normalizer = np.sum(compositions,axis=1).values.reshape(-1,1)
compositions = compositions/normalizer

#Ideal mixing entropy - Sid
    #Sid = Ideal mixing entropy 
Sid = -(compositions*np.log(compositions)).sum(axis=1) 

#Electronegativity
    #Electronegativity_avgs_w = Electronegativity weighted avarage
    #Electronegativity_std_devs_w = Electronegativity weighted standard deviation
Electronegativity = properties['Electronegativity']
Electronegativity_avgs_w = []
Electronegativity_std_devs_w = []
for idx, row in compositions.iterrows():
    weights = row.values
    Electronegativity_avg_w = np.average(Electronegativity, weights=weights) 
    Electronegativity_std_dev_w = np.sqrt(np.average((Electronegativity-Electronegativity_avg_w)**2, weights=weights))
    Electronegativity_avgs_w.append(Electronegativity_avg_w)
    Electronegativity_std_devs_w.append(Electronegativity_std_dev_w)

#Valence Electron Concentration - VEC
    #VEC_avgs_w = Valence Electron Concentration weighted avarage
    #VEC_std_devs_w = Valence Electron Concentration weighted standard deviation
VEC = properties['NValence']
VEC_avgs_w = []
VEC_std_devs_w = []
for idx, row in compositions.iterrows():
    weights = row.values
    VEC_avg_w = np.average(VEC, weights=weights) 
    VEC_std_dev_w = np.sqrt(np.average((VEC-VEC_avg_w)**2, weights=weights))
    VEC_avgs_w.append(VEC_avg_w)
    VEC_std_devs_w.append(VEC_std_dev_w)
    
#Atomic Radius
    #AtomicRadius_avgs_w = Atomic Radius weighted avarage
    #AtomicRadius_std_devs_w = Atomic Radius weighted standard deviation
AtomicRadius = properties['AtomicRadius']
AtomicRadius_avgs_w = []
AtomicRadius_std_devs_w = []
for idx, row in compositions.iterrows():
    weights = row.values
    AtomicRadius_avg_w = np.average(AtomicRadius, weights=weights) 
    AtomicRadius_std_dev_w = np.sqrt(np.average((AtomicRadius-AtomicRadius_avg_w)**2, weights=weights))
    AtomicRadius_avgs_w.append(AtomicRadius_avg_w)
    AtomicRadius_std_devs_w.append(AtomicRadius_std_dev_w)

#Lattice Constant
    #LatticeConstant_avgs_w = Lattice Constant weighted avarage
    #LatticeConstant_std_devs_w = Lattice Constant weighted standard deviation
LatticeConstant = properties['LatticeConstant']
LatticeConstant_avgs_w = []
LatticeConstant_std_devs_w = []
for idx, row in compositions.iterrows():
    weights = row.values
    LatticeConstant_avg_w = np.average(LatticeConstant, weights=weights) 
    LatticeConstant_std_dev_w = np.sqrt(np.average((LatticeConstant-LatticeConstant_avg_w)**2, weights=weights))
    LatticeConstant_avgs_w.append(LatticeConstant_avg_w)
    LatticeConstant_std_devs_w.append(LatticeConstant_std_dev_w)
    
#Melting Temperature - MeltT
    #MeltT_avgs_w = Melting Temperature weighted avarage
    #MeltT_std_devs_w = Melting Temperature weighted standard deviation
MeltT = properties['MeltT']
MeltT_avgs_w = []
MeltT_std_devs_w = []
for idx, row in compositions.iterrows():
    weights = row.values
    MeltT_avg_w = np.average(MeltT, weights=weights) 
    MeltT_std_dev_w = np.sqrt(np.average((MeltT-MeltT_avg_w)**2, weights=weights))
    MeltT_avgs_w.append(MeltT_avg_w)
    MeltT_std_devs_w.append(MeltT_std_dev_w)

#Meq
    #Meq_avgs_w = Meq weighted avarage
    #Meq_std_devs_w = Meq weighted standard deviation
Meq = properties['Meq']
Meq_avgs_w = []
Meq_std_devs_w = []
for idx, row in compositions.iterrows():
    weights = row.values
    Meq_avg_w = np.average(Meq, weights=weights)
    Meq_std_dev_w = np.sqrt(np.average((Meq-Meq_avg_w)**2, weights=weights))
    Meq_avgs_w.append(Meq_avg_w)
    Meq_std_devs_w.append(Meq_std_dev_w)

#Md
    #Md_avgs_w = Md weighted avarage
    #Md_std_devs_w = Md weighted standard deviation
Md = properties['Md']
Md_avgs_w = []
Md_std_devs_w = []
for idx, row in compositions.iterrows():
    weights = row.values
    Md_avg_w = np.average(Md, weights=weights)
    Md_std_dev_w = np.sqrt(np.average((Md-Md_avg_w)**2, weights=weights))
    Md_avgs_w.append(Md_avg_w)
    Md_std_devs_w.append(Md_std_dev_w)

#Bo
    #Bo_avgs_w = Bo weighted avarage
    #Bo_std_devs_w = Bo weighted standard deviation
Bo = properties['Md']
Bo_avgs_w = []
Bo_std_devs_w = []
for idx, row in compositions.iterrows():
    weights = row.values
    Bo_avg_w = np.average(Bo, weights=weights)
    Bo_std_dev_w = np.sqrt(np.average((Bo-Bo_avg_w)**2, weights=weights))
    Bo_avgs_w.append(Bo_avg_w)
    Bo_std_devs_w.append(Bo_std_dev_w)

#PHI
arrAtomicSize = np.array(properties['AtomicRadius']) * 100
arrMeltingT = np.array(properties['MeltT'])
elements = np.array(properties['element'])

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

alloys = np.asarray(compositions) / 100
phiAlloys = parPhi(alloys)

#Df creation
descriptors = {
    'Cr': database['Cr'],
    'Cu': database['Cu'],
    'Fe': database['Fe'],
    'Mn': database['Mn'],
    'Mo': database['Mo'],
    'Nb': database['Nb'],
    'Si': database['Si'],
    'Sn': database['Sn'],
    'Ta': database['Ta'],
    'Ti': database['Ti'],
    'V': database['V'],
    'Zr': database['Zr'],
    #'E': database['E'],
    'Sid': Sid,
    'Electronegativity_avgs_w' : Electronegativity_avgs_w,
    'Electronegativity_std_devs_w' : Electronegativity_std_devs_w,
    'VEC_avgs_w' : VEC_avgs_w,
    'VEC_std_devs_w' : VEC_std_devs_w,
    'AtomicRadius_avgs_w' : AtomicRadius_avgs_w,
    'AtomicRadius_std_devs_w': AtomicRadius_std_devs_w,
    'LatticeConstant_avgs_w' : LatticeConstant_avgs_w,
    'LatticeConstant_std_devs_w' : LatticeConstant_std_devs_w,
    'MeltT_avgs_w' : MeltT_avgs_w,
    'MeltT_std_devs_w' : MeltT_std_devs_w,
    'phi': phiAlloys,
    'Meq_avgs_w': Meq_avgs_w,
    'Meq_std_devs_w': Meq_std_devs_w,
    'Md_avgs_w':  Md_avgs_w,
    'Md_std_devs_w': Md_std_devs_w,
    'Bo_avgs_w': Bo_avgs_w,
    'Bo_std_devs_w': Bo_std_devs_w
    }
E_ML=pd.DataFrame(descriptors)
E_ML.to_csv('E_ML_descriptors.csv',index=False)