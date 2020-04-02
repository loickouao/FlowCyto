from minisom import MiniSom
import numpy as np
from LoadFcs import *
import sys
import matplotlib.pyplot as plt 

mypath = "../inst/extdata/21-10-15_Tube_011_Live.fcs"
ff = Loadfcs(mypath)
ff_comp = ff.compensate()
data = ff.transform(ff_comp, range(7,20), type_transform="logicle")

idx_interest = [7] + list(range(9,14)) + list(range(15,20))
d = len(idx_interest)
data = data[:, idx_interest]

som = MiniSom(10, 10, d, sigma=2.5, learning_rate=0.05, 
              neighborhood_function='gaussian', random_seed=1)
som.pca_weights_init(data)
max_iter = 140000

q_error = []
t_error = []
iter_x = []
for i in range(max_iter):
    percent = 100*(i+1)/max_iter
    rand_i = np.random.randint(len(data)) # This corresponds to train_random() method.
    som.update(data[rand_i], som.winner(data[rand_i]), i, max_iter)
    if (i+1) % 1000 == 0:
        q_error.append(som.quantization_error(data))
        t_error.append(som.topographic_error(data))
        iter_x.append(i)
        sys.stdout.write(f'\riteration={i:2d} status={percent:0.2f}% \n')   
fig, ax = plt.subplots()
ax.plot(iter_x, q_error, 'b', label='q_error')
ax.plot(iter_x, t_error, 'r--',  label='t_error')
ax.set_label('quantization error')
ax.set_label('iteration index')
ax.set_title('Quantization error and topographic error by iteration')
leg = ax.legend()
plt.savefig("Resultats/4_Quantization_error/Figure_1 - Quantization and topographic error by iteration.png")
plt.show