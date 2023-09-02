import os
from copy import deepcopy
import numpy as np
from WSN import WSN_fit
from scipy.stats import levy
from pyDOE2 import lhs


PopSize = 10                                                  # the number of individuals (PopSize > 4)
DimSize = 10                                                    # the number of variables
LB = [-100] * DimSize
UB = [100] * DimSize                                                 # the minimum value of the variable range
Trials = 30                                                   # the number of independent runs
MaxFEs = DimSize * 1000                      # the maximum number of fitness evaluations
GC = 6                                                                # the maximum growth cycle of an individual
GR = 1                                                                # the maximum growth radius of an individual
MS = 2                                                                # the moving scale
SeedNum = 6                                                          # the number of generated seeds by each individual

Pop = np.zeros((PopSize, DimSize))               # the coordinates of the individual (candidate solutions)
PopFit = np.zeros(PopSize)                        # the fitness value of all individuals
curSpan = 0
curFEs = 0                                                              # the current number of fitness evaluations
SuiteName = "CEC2013"

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
        PopFit[i] = -func(Pop[i])
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
    Off = np.zeros((PopSize, DimSize))
    OffFit = np.zeros(PopSize)
    for i in range(PopSize):
        Off[i], index = Exploitation(i)
        CheckIndi(Off[i])
        OffFit[i] = -func(Off[i])
        Times_table[index] += 1
        Temp_table[index] += PopFit[i] - OffFit[i]
        curFEs += 1
        if OffFit[i] < PopFit[i]:
            PopFit[i] = OffFit[i]
            Pop[i] = Off[i].copy()
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
    seedIndi = np.zeros((PopSize*SeedNum, DimSize))
    seedFit = np.zeros(PopSize*SeedNum)
    Temp_table = np.zeros(4)
    Times_table = [0.00000000001] * 4
    for i in range(PopSize):
        for j in range(SeedNum):
            seedIndi[i*SeedNum + j], index = Exploration(i)
            CheckIndi(seedIndi[i*SeedNum + j])
            seedFit[i*SeedNum + j] = -func(seedIndi[i*SeedNum + j])
            Times_table[index] += 1
            Temp_table[index] += seedFit[i*SeedNum + j] - PopFit[i]
            curFEs += 1
    for i in range(len(Temp_table)):
        Temp_table[i] /= Times_table[i]
    Temp_table *= Gamma
    for i in range(len(Temp_table)):
        Q_table_exploration[i] += Temp_table[i]

    temp_individual = np.vstack((Pop, seedIndi))
    temp_individual_fitness = np.hstack((PopFit, seedFit))
    tmp = list(map(list, zip(range(len(temp_individual_fitness)), temp_individual_fitness)))
    small = sorted(tmp, key=lambda x: x[1], reverse=False)
    for i in range(PopSize):
        key, _ = small[i]
        PopFit[i] = temp_individual_fitness[key]
        Pop[i] = temp_individual[key].copy()


def VegetationEvolution(bench):
    global curSpan, GC, curFEs
    if curSpan < GC:
        Growth(bench)
        curSpan += 1
    elif curSpan == GC:
        Maturity(bench)
        curSpan = 0
    else:
        print("Error: Maximum generation period exceeded.")


def RunVEGE(func):
    global curFEs, FuncNum, Pop, PopFit, SuiteName, Gamma, Q_table_exploration, Q_table_exploitation
    All_Trial_Best = []
    All_Best = []
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
            VegetationEvolution(func)
            Best_list.append(min(PopFit))
            if curFEs % 120 == 0:
                Gamma *= 0.9
        All_Trial_Best.append(np.abs(Best_list))
        All_Best.append(Pop[np.argmax(PopFit)])
    np.savetxt("./QVEGE_Data/WSN/Obj/WSN_" + str(int(Dim / 2)) + ".csv", All_Trial_Best, delimiter=",")
    np.savetxt("./QVEGE_Data/WSN/Sol/WSN_" + str(int(Dim / 2)) + ".csv", All_Best, delimiter=",")


def main_WSN(Dim):
    global DimSize, Pop, MaxFEs, SuiteName, LB, UB
    DimSize = Dim
    Pop = np.zeros((PopSize, DimSize))
    MaxFEs = 3000
    LB = [0] * DimSize
    UB = [50] * DimSize

    RunVEGE(WSN_fit)


if __name__ == "__main__":
    if os.path.exists('./QVEGE_Data/WSN/Obj') == False:
        os.makedirs('./QVEGE_Data/WSN/Obj')
    if os.path.exists('./QVEGE_Data/WSN/Sol') == False:
        os.makedirs('./QVEGE_Data/WSN/Sol')
    Dims = [64, 84, 108]
    for Dim in Dims:
        main_WSN(Dim)
