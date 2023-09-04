## Introductory notebooks for the ParaDiag parallel-in-time method

Scripts for solving an ODE (Dahlquist's equation) and a PDE (linear advection-diffusion) with ParaDiag.
Each directory contains the following files:

- `serial.py` a plain python script for solving the serial in time method.
- `serial.ipynb` an incomplete ipython notebook for the serial in time method meant to be used as an exercise.
- `serial-completed.ipynb` the completed ipython notebook for the serial in time method.
- `paradiag.py` a plain python script for solving the paradiag method.
- `paradiag.ipynb` an incomplete ipython notebook for the paradiag method meant to be used as an exercise.
- `paradiag-completed.ipynb` the completed ipython notebook for the paradiag method.

It is recommended to start with the `serial.ipynb` notebook as this will introduce the equation and the base time-integration method.
The `serial-completed.ipynb` notebook can be used to check solutions.
The `serial.py` script can be used to more easily play around with different parameters once the notebooks are completed.

Once the serial-in-time exercise is completed, the `paradiag.*` files follow the same pattern.
The`paradiag.ipynb` introduces the paradiag method for each equation.
The `paradiag-completed.ipynb` notebook can be used to check solutions.
The `paradiag.py` script can be used to try out different parameter combinations.
