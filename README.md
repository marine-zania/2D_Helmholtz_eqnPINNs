## PINNs for Rectangular Waveguide Helmholtz Equation

This project implements a Physics-Informed Neural Network (PINN) in PyTorch to solve the 2D Helmholtz equation for TEmn

​modes in a rectangular waveguide with Dirichlet boundary conditions.



Features

Solves 

&nbsp;∇²H\_z + k² H\_z = 0



with Dirichlet boundary conditions:H\_z = 0  on all boundaries on all boundaries

Computes and visualizes multiple TEmn modes



Modular code, easy to modify for different modes



----

###### Roadmap

Current: Rectangular waveguide



Upcoming: PINNs for slanted (trapezoidal), circular, and elliptical waveguides



------

###### Possible Improvements

* Make k2 a learnable parameter (eigenvalue discovery)



* Add support for more general domains



* Improve mode isolation and convergence speed



This repository will be updated with more geometries and improvements. Contributions are welcome!

