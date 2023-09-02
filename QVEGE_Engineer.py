#coding:UTF-8
'''
Created by Jun YU (yujun@ie.niigata-u.ac.jp) on November 22, 2022
benchmark function: 28 functions of the CEC2017 test suite (https://www3.ntu.edu.sg/home/epnsugan/index_files/CEC2017/CEC2017.htm)
reference paper: Jun Yu, "Vegetation Evolution: An Optimization Algorithm Inspired by the Life Cycle of Plants," 
                          International Journal of Computational Intelligence and Applications, vol. 21, no.2, Article No. 2250010
'''
import os
from enoppy.paper_based.pdo_2022 import *
from copy import deepcopy
import numpy as np
from scipy.stats import levy
from pyDOE2 import lhs

PopSize = 10                                                  # the number of individuals (PopSize > 4)
DimSize = 10                                                    # the number of variables
LB = [-100] * DimSize
UB = [100] * DimSize                                                 # the minimum value of the variable range
Trials = 30                                                   # the number of independent runs
MaxFEs = DimSize * 500                      # the maximum number of fitness evaluations
GC = 6                                                                # the maximum growth cycle of an individual
GR = 1                                                                # the maximum growth radius of an individual
MS = 2                                                                # the moving scale
SeedNum = 6                                                          # the number of generated seeds by each individual

Pop = np.zeros((PopSize, DimSize))               # the coordinates of the individual (candidate solutions)
PopFit = np.zeros(PopSize)                        # the fitness value of all individuals
curSpan = 0
curFEs = 0                                                              # the current number of fitness evaluations
FuncNum = None
SuiteName = "Engineer"
Q_table_exploration = np.zeros(4)
Q_table_exploitation = np.zeros(4)
epsilon = 0.5
Gamma = 1


def CheckIndi(Indi):
    for i in range(DimSize):
        range_width = UB[i] - LB[i]
        if Indi[i] > UB[i]:
            n = int((Indi[i] - UB[i]) / range_width)
            mirrorRange = (Indi[i] - UB[i]) - (n * range_width)
            Indi[i] = UB[i] - mirrorRange
        elif Indi[i] < LB[i]:
            n = int((LB[i] - Indi[i]) / range_width)
            mirrorRange = (LB[i] - Indi[i]) - (n * range_width)
            Indi[i] = LB[i] + mirrorRange
        else:
            pass


# initialize the Pop randomly
def Initialization(func):
    global Pop, PopFit, curFEs, curSpan
    Pop = lhs(DimSize, samples=PopSize)
    for i in range(PopSize):
        for j in range(DimSize):
            Pop[i][j] = LB[j] + (UB[j] - LB[j]) * Pop[i][j]
        PopFit[i] = func(Pop[i])
        curFEs += 1
    curSpan = 1


def LocalSearch(indi):
    global GR, DimSize
    off = deepcopy(indi)
    for i in range(DimSize):
        off[i] += GR * (np.random.random() * 2.0 - 1.0)
    return off


def NormalSearch(indi):
    global GR, DimSize
    off = deepcopy(indi)
    for i in range(DimSize):
        off[i] += GR * (np.random.normal(0, 1))
    return off


def LevySearch(indi):
    global DimSize
    off = deepcopy(indi)
    for i in range(DimSize):
        off[i] += levy.rvs()
    return off


def ChebyshevMap():
    global GR, DimSize
    v = np.zeros(DimSize)
    v[0] = np.random.rand()
    for i in range(1, DimSize):
        v[i] = np.cos(i / np.cos(v[i-1]))
    return GR * v


def ChaosSearch(indi):
    global GR, DimSize
    off = deepcopy(indi)
    off += ChebyshevMap()
    return off


def isZero(Q_table):
    for i in Q_table:
        if i != 0:
            return False
    return True


def epsilonGreedy(Q_table):
    global epsilon
    size = len(Q_table)
    if isZero(Q_table):
        return np.random.randint(0, size)
    else:
        if np.random.rand() < epsilon:
            return np.argmax(Q_table)
        else:
            return np.random.randint(0, size)


def Exploitation(i):
    global Pop, Q_table_exploitation
    archive = [LocalSearch, NormalSearch, LevySearch, ChaosSearch]
    index = epsilonGreedy(Q_table_exploitation)
    strategy = archive[index]
    return strategy(Pop[i]), index


def Growth(func):
    global Pop, PopFit, curFEs, Q_table_exploitation, Gamma
    Temp_table = np.zeros(4)
    Times_table = [0.00000000001] * 4
    offspring = np.zeros((PopSize, DimSize))
    offspring_fitness = np.zeros(PopSize)
    for i in range(PopSize):
        offspring[i], index = Exploitation(i)
        CheckIndi(offspring[i])
        offspring_fitness[i] = func(offspring[i])
        Times_table[index] += 1
        Temp_table[index] += PopFit[i] - offspring_fitness[i]
        curFEs += 1
        if offspring_fitness[i] < PopFit[i]:
            PopFit[i] = offspring_fitness[i]
            Pop[i] = offspring[i].copy()
    for i in range(len(Temp_table)):
        Temp_table[i] /= Times_table[i]
    Temp_table *= Gamma
    for i in range(len(Temp_table)):
        Q_table_exploitation[i] += Temp_table[i]


def Cur(i):
    global Pop, MS, PopSize
    candi = list(range(0, PopSize))
    candi.remove(i)
    r1, r2 = np.random.choice(candi, 2, replace=False)
    return Pop[i] + MS * (np.random.random() * 2.0 - 1.0) * (Pop[r1] - Pop[r2])


def CurToRand(i):
    global Pop, MS, PopSize
    candi = list(range(0, PopSize))
    candi.remove(i)
    r1, r2, r3 = np.random.choice(candi, 3, replace=False)
    return Pop[i] + MS * (np.random.random() * 2.0 - 1.0) * (Pop[r1] - Pop[i]) + MS * (np.random.random() * 2.0 - 1.0) * (Pop[r2] - Pop[r3])


def CurToBest(i):
    global Pop, PopFit, MS, PopSize
    X_best = Pop[np.argmin(PopFit)]
    candi = list(range(0, PopSize))
    candi.remove(i)
    r1, r2 = np.random.choice(candi, 2, replace=False)
    return Pop[i] + MS * (np.random.random() * 2.0 - 1.0) * (X_best - Pop[i]) + MS * (np.random.random() * 2.0 - 1.0) * (Pop[r1] - Pop[r2])


def CurTopBest(i):
    global Pop, PopFit, MS, PopSize, DimSize
    p = 0.5
    size = int(p * PopSize)
    index = np.argsort(PopFit)[0:size]
    X_pbest = np.zeros(DimSize)
    for j in index:
        X_pbest += Pop[j]
    X_pbest /= size
    candi = list(range(0, PopSize))
    candi.remove(i)
    r1, r2 = np.random.choice(candi, 2, replace=False)
    return Pop[i] + MS * (np.random.random() * 2.0 - 1.0) * (X_pbest - Pop[i]) + MS * (np.random.random() * 2.0 - 1.0) * (Pop[r1] - Pop[r2])


def Exploration(i):
    global Pop, Q_table_exploration
    archive = [Cur, CurToRand, CurToBest, CurTopBest]
    index = epsilonGreedy(Q_table_exploration)
    strategy = archive[index]
    return strategy(i), index


def Maturity(func):
    global Pop, PopFit, curFEs, Q_table_exploration
    seed_individual = np.zeros((PopSize*SeedNum, DimSize))
    seed_individual_fitness = np.zeros(PopSize*SeedNum)
    Temp_table = np.zeros(4)
    Times_table = [0.00000000001] * 4
    for i in range(PopSize):
        for j in range(SeedNum):
            seed_individual[i*SeedNum + j], index = Exploration(i)
            CheckIndi(seed_individual[i*SeedNum + j])
            seed_individual_fitness[i*SeedNum + j] = func(seed_individual[i*SeedNum + j])
            Times_table[index] += 1
            Temp_table[index] += seed_individual_fitness[i*SeedNum + j] - PopFit[i]
            curFEs += 1
    for i in range(len(Temp_table)):
        Temp_table[i] /= Times_table[i]
    Temp_table *= Gamma
    for i in range(len(Temp_table)):
        Q_table_exploration[i] += Temp_table[i]

    tmpIndi = np.vstack((Pop, seed_individual))
    tmpFit = np.hstack((PopFit, seed_individual_fitness))
    tmp = list(map(list, zip(range(len(tmpFit)), tmpFit)))
    small = sorted(tmp, key=lambda x: x[1], reverse=False)
    for i in range(PopSize):
        key, _ = small[i]
        PopFit[i] = tmpFit[key]
        Pop[i] = tmpIndi[key].copy()


def QVEGE(func):
    global curSpan, GC, curFEs
    if curSpan < GC:
        Growth(func)
        curSpan += 1
    elif curSpan == GC:
        Maturity(func)
        curSpan = 0
    else:
        print("Error: Maximum generation period exceeded.")


def RunQVEGE(func):
    global curFEs, FuncNum, PopFit, SuiteName, Gamma, Q_table_exploration, Q_table_exploitation
    All_Trial_Best = []
    for i in range(Trials):
        Best_list = []
        curFEs = 0
        Gamma = 1
        Q_table_exploration = np.zeros(4)
        Q_table_exploitation = np.zeros(4)
        np.random.seed(2022 + 88 * i)
        Initialization(func)
        Best_list.append(min(PopFit))
        while curFEs < MaxFEs:
            QVEGE(func)
            Best_list.append(min(PopFit))
            if curFEs % 120 == 0:
                Gamma *= 0.9
        All_Trial_Best.append(Best_list)
    np.savetxt('./QVEGE_Data/Engineer/' + FuncNum + '.csv', All_Trial_Best, delimiter=",")


def main():
    global FuncNum, DimSize, Pop, MaxFEs, SuiteName, LB, UB
    Probs = [WBP(), PVP(), CSP(), SRD(), TBTD(), GTD(), CBD(), IBD(), TCD(), PLD(), CBHD(), RCB()]
    Names = ["WBP", "PVP", "CSP", "SRD", "TBTD", "GTD", "CBD", "IBD", "TCD", "PLD", "CBHD", "RCB"]
    MaxFEs = 20000
    
    for i in range(len(Probs)):
        DimSize = Probs[i].n_dims
        LB = Probs[i].lb
        UB = Probs[i].ub
        Pop = np.zeros((PopSize, DimSize))
        FuncNum = Names[i]
        RunQVEGE(Probs[i].evaluate)


if __name__ == "__main__":
    if os.path.exists('./QVEGE_Data/Engineer') == False:
        os.makedirs('./QVEGE_Data/Engineer')
    main()
