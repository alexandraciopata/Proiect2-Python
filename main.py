import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from functii import *

t = pd.read_csv("C:\\Users\\MateBook D15\\Downloads\\proiect_dsad_ex2\\date_salariati.csv", index_col=1)
nan_replace(t)

variabile_observate = list(t)[2:]

# standardizarea datelor
x = (t[variabile_observate] - np.mean(t[variabile_observate], axis=0)) / np.std(t[variabile_observate], axis=0)
n, m = x.shape

# apel PCA peste setul de date
model_acp = PCA()
model_acp.fit(x)


alpha = model_acp.explained_variance_
a = model_acp.components_
c = model_acp.transform(x)
print("alpha", alpha)
print("a", a)
print("c", c)

etichete = ["C" + str(i + 1) for i in range(len(alpha))]

comp_tabelar = tabelare_matrice(c, t.index, etichete, "componente.csv")
plot_componente(comp_tabelar, "C1", "C2")


# Kaiser
where = np.where(alpha > 1)
nr_comp_kaiser = len(where[0])
print("Comp principale cf crit. Kaiser: ", nr_comp_kaiser)

# procent de acoperire
ponderi = np.cumsum(alpha / sum(alpha))
where = np.where(ponderi > 0.8)
nr_comp_procent = where[0][0] + 1
print("Comp principale cf crit. Procent de acoperire: ", nr_comp_procent)

# Cattell
eps = alpha[:(m - 1)] - alpha[1:]
sigma = eps[:(m - 2)] - eps[1:]
negative = sigma < 0
if any(negative):
    where = np.where(negative)
    nr_comp_cattell = where[0][0] + 1
else:
    nr_comp_cattell = None
print("Comp principale cf crit. Cattell: ", nr_comp_cattell)

# Calcul corelatii intre componente principale si variabile observate
corr = np.corrcoef(x, c, rowvar=False)
r_x_c =  corr[:m, :m]
r_x_c_tabelar = tabelare_matrice(r_x_c, variabile_observate, etichete, "corelatii_factoriale.csv")

corelograma(r_x_c_tabelar)
plot_corelatii(r_x_c_tabelar, "C1", "C2")
plot_corelatii(r_x_c_tabelar, "C1", "C3")

# calcul cosinusuri
comp_patrat = c * c
cosin = np.transpose(comp_patrat.T / np.sum(comp_patrat, axis=1))
consi_tabelar = tabelare_matrice(cosin, t.index, etichete, "cosin.csv")

# calcul comunalitati
r_x_c_patrat = r_x_c * r_x_c
comunalitati = np.cumsum(r_x_c_patrat, axis=1)
comunalitati_tabelar = tabelare_matrice(comunalitati, variabile_observate,
                                        etichete, "comunalitati.csv")

corelograma(comp_tabelar)
afisare()