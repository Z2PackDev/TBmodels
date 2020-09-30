#!/usr/bin/env python

# (c) 2015-2018, ETH Zurich, Institut fuer Theoretische Physik
# Author: Dominik Gresch <greschd@gmx.ch>

import tbmodels
import numpy as np
import matplotlib.pyplot as plt

MODEL_TB = tbmodels.io.load("data/tb_model.hdf5")

if __name__ == "__main__":
    k_star = np.array([0.1, 0.2, 0.3])
    k_dir = np.array([0.3, -0.1, 0.1])
    model_kp = MODEL_TB.construct_kdotp(k_star, order=2)

    bands_tb = np.array(
        [MODEL_TB.eigenval(k_star + x * k_dir) for x in np.linspace(-1, 1, 100)]
    )
    bands_kp = np.array([model_kp.eigenval(x * k_dir) for x in np.linspace(-1, 1, 100)])

    plt.plot(bands_tb, color="C0")
    plt.plot(bands_kp, color="C1")
    plt.ylim(np.min(bands_tb) - 0.5, np.max(bands_tb) + 0.5)
    plt.show()
