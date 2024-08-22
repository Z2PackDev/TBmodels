#!/usr/bin/env python
# Construct a Model from wannier90 tb.dat file.
import os
import shutil
import subprocess
import numpy as np
import tbmodels as tb
import matplotlib.pyplot as plt

if __name__ == "__main__":
    WANNIER90_COMMAND = os.path.expanduser("~/git/wannier90/wannier90.x")
    BUILD_DIR = "./build_tb"

    if not os.path.exists(BUILD_DIR):
        shutil.copytree("./input", BUILD_DIR)
        subprocess.call([WANNIER90_COMMAND, "silicon"], cwd=BUILD_DIR)

    model = tb.Model.from_wannier_tb_files(
        tb_file=f"{BUILD_DIR}/silicon_tb.dat",
        wsvec_file=f"{BUILD_DIR}/silicon_wsvec.dat",
    )
    print(model)

    # Compute band structure along an arbitrary kpath
    theta = 37 / 180 * np.pi
    phi = 43 / 180 * np.pi
    rlist = np.linspace(0, 2, 20)
    klist = [
        [
            r * np.sin(theta) * np.cos(phi),
            r * np.sin(theta) * np.sin(phi),
            r * np.cos(theta),
        ]
        for r in rlist
    ]

    eigvals = model.eigenval(klist)

    plt.plot(eigvals)
    plt.show()
