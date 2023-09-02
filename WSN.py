import numpy as np
import matplotlib.pyplot as plt



def Euclid(X, Y):
    return (X[0] - Y[0]) ** 2 + (X[1] - Y[1]) ** 2


def WSN_fit(indi, scale=(0, 50), radius=5, gran=1):
    radius_pow = radius * radius
    lb, ub = scale[0], scale[1]
    pos = np.array(indi).reshape(-1, 2)
    sum_sensor = 0
    inner = 0
    for i in np.arange(lb, ub, gran):
        for j in np.arange(lb, ub, gran):
            sum_sensor += 1
            for particle in pos:
                if Euclid(particle, [i, j]) <= radius_pow:
                    inner += 1
                    break
    return inner / sum_sensor


def Draw_indi(indi, title, name, scale=(0, 50), radius=5):
    plt.xlim(scale[0], scale[1])
    plt.ylim(scale[0], scale[1])
    pos = np.array(indi).reshape(-1, 2)

    plt.title(title)
    ax = plt.gca()
    for particle in pos:
        ax.add_patch(plt.Circle((particle[0], particle[1]), radius=radius, facecolor='lightcyan', edgecolor="r")) # 'lightyellow', 'lightcyan'
    for particle in pos:
        ax.scatter(particle[0], particle[1], color='black')
    plt.savefig("./pics/" + name, dpi=750)
    plt.show()


# indi = np.random.uniform(0, 50, 64)
# scale = [0, 50]
# # print(WSN_fit(indi))
# Draw_indi(indi, scale, name="", radius=5)



