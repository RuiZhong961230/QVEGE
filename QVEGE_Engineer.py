import os
from copy import deepcopy
import numpy as np
from enoppy.paper_based.pdo_2022 import *
from scipy.stats import levy
from pyDOE2 import lhs

PopSize = 10
DimSize = 10
LB = [-100] * DimSize
UB = [100] * DimSize
Trials = 3
MaxFEs = DimSize * 1000
GC = 6  # Maximum Growth Cycle
GR = 1  # Maximum Growth Radius
MS = 2  # Moving Scale
SeedNum = 6  # Number of generated seeds by each individual

Pop = np.zeros((PopSize, DimSize))
PopFit = np.zeros(PopSize)
curSpan = 0
curFEs = 0
SuiteName = "CEC2013"

Q_table_exploitation = np.zeros((PopSize, 4))
Q_table_exploration = np.zeros((PopSize, 4))
alpha = 0.1  # Learning Rate
epsilon_initial = 0.9  # Initial exploration rate (Increased for more initial exploration)
epsilon_final = 0.05
epsilon_decay_steps = MaxFEs // 2  # Epsilon decays over half the max FEs
Gamma = 0.95  # Discount factor (standard Q-learning gamma)


def CheckIndi(Indi):
    # Boundary handling remains the same (reflecting boundary)
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
        PopFit[i] = func.evaluate(Pop[i])
        curFEs += 1
    curSpan = 1

# --- Exploitation Strategies (Growth) ---
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
        v[i] = np.cos(i / np.cos(v[i - 1]))
    return GR * v


def ChaosSearch(indi):
    global GR, DimSize
    off = deepcopy(indi)
    off += ChebyshevMap()
    return off


# --- Exploration Strategies (Maturity) ---
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
    return Pop[i] + MS * (np.random.random() * 2.0 - 1.0) * (Pop[r1] - Pop[i]) + MS * (
                np.random.random() * 2.0 - 1.0) * (Pop[r2] - Pop[r3])


def CurToBest(i):
    global Pop, PopFit, MS, PopSize
    X_best = Pop[np.argmin(PopFit)]
    candi = list(range(0, PopSize))
    candi.remove(i)
    r1, r2 = np.random.choice(candi, 2, replace=False)
    return Pop[i] + MS * (np.random.random() * 2.0 - 1.0) * (X_best - Pop[i]) + MS * (
                np.random.random() * 2.0 - 1.0) * (Pop[r1] - Pop[r2])


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
    return Pop[i] + MS * (np.random.random() * 2.0 - 1.0) * (X_pbest - Pop[i]) + MS * (
                np.random.random() * 2.0 - 1.0) * (Pop[r1] - Pop[r2])


# --- Q-Learning Functions  ---
def get_current_epsilon():
    global curFEs, epsilon_initial, epsilon_final, epsilon_decay_steps
    if curFEs >= epsilon_decay_steps:
        return epsilon_final
    return epsilon_initial - (epsilon_initial - epsilon_final) * (curFEs / epsilon_decay_steps)


def epsilonGreedy(Q_table_i):
    # Q_table_i is the Q-vector for individual i
    epsilon = get_current_epsilon()
    size = len(Q_table_i)
    if np.random.rand() < epsilon:
        return np.random.randint(0, size)
    else:
        max_indices = np.where(Q_table_i == np.max(Q_table_i))[0]
        return np.random.choice(max_indices)


def Exploitation(i):
    global Pop, Q_table_exploitation
    archive = [LocalSearch, NormalSearch, LevySearch, ChaosSearch]
    index = epsilonGreedy(Q_table_exploitation[i])
    strategy = archive[index]
    return strategy(Pop[i]), index


def Exploration(i):
    global Pop, Q_table_exploration
    archive = [Cur, CurToRand, CurToBest, CurTopBest]
    index = epsilonGreedy(Q_table_exploration[i])
    strategy = archive[index]
    return strategy(i), index


def Growth(func):
    global Pop, PopFit, curFEs, Q_table_exploitation, alpha, Gamma
    Off = np.zeros((PopSize, DimSize))
    OffFit = np.zeros(PopSize)

    # 1. Generate Offspring and Evaluate
    for i in range(PopSize):
        # Select strategy using epsilon-greedy based on Pop[i]'s Q-values
        Off[i], index = Exploitation(i)
        CheckIndi(Off[i])
        OffFit[i] = func.evaluate(Off[i])
        curFEs += 1

        # 2. Calculate Reward
        # Reward is the fitness improvement (positive is good)
        reward = PopFit[i] - OffFit[i]

        # 3. Q-Table Update (Classic SARSA-style update without next state estimation)
        # Q(s, a) <- Q(s, a) + alpha * [reward - Q(s, a)]
        # Since we use strategies for the *current* state, we simplify the update.
        Q_table_exploitation[i, index] += alpha * (reward - Q_table_exploitation[i, index])

        # 4. Survival selection
        if OffFit[i] < PopFit[i]:
            PopFit[i] = OffFit[i]
            Pop[i] = Off[i].copy()


def Maturity(func):
    global Pop, PopFit, curFEs, Q_table_exploration, alpha, Gamma
    seedIndi = np.zeros((PopSize * SeedNum, DimSize))
    seedFit = np.zeros(PopSize * SeedNum)

    # 1. Generate Seeds and Evaluate
    for i in range(PopSize):
        for j in range(SeedNum):
            # i is the parent index, k is the index in the seed array
            k = i * SeedNum + j

            # Select strategy using epsilon-greedy based on Pop[i]'s Q-values
            seedIndi[k], index = Exploration(i)
            CheckIndi(seedIndi[k])
            seedFit[k] = func.evaluate(seedIndi[k])
            curFEs += 1

            # 2. Calculate Reward
            # The reward is based on the quality of the generated seed.
            # Here, we use the fitness difference, but the context is exploration.
            # A common approach is rewarding successful exploration (creating a better-than-parent seed).
            # Reward = 1 if seed is better, 0 otherwise. Let's stick to the difference for consistency.
            reward = PopFit[i] - seedFit[k]

            # 3. Q-Table Update (Individual update for parent i)
            # We use the average reward of the seeds generated by the parent i using strategy 'index'
            # to update the Q-value for that strategy for parent i.
            Q_table_exploration[i, index] += alpha * (reward - Q_table_exploration[i, index])

    # 4. Selection (Population Update)
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


def RunQVEGE(func):
    global curFEs, FuncNum, PopFit, SuiteName, Q_table_exploration, Q_table_exploitation
    global PopSize, Q_table_exploitation, Q_table_exploration
    All_Trial_Best = []

    for i in range(Trials):
        Best_list = []
        curFEs = 0

        # Reset Q-tables for each trial, now with PopSize rows
        Q_table_exploitation = np.zeros((PopSize, 4))
        Q_table_exploration = np.zeros((PopSize, 4))

        np.random.seed(2022 + 88 * i)
        Initialization(func)

        Best_list.append(min(PopFit))
        while curFEs < MaxFEs:
            VegetationEvolution(func)
            Best_list.append(min(PopFit))
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
        RunQVEGE(Probs[i])


if __name__ == "__main__":
    if os.path.exists('./QVEGE_Data/Engineer') == False:
        os.makedirs('./QVEGE_Data/Engineer')
    main()

