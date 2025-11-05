# Accelerated Lebwohl–Lasher Simulation

This project implements and compares several optimisation and parallelisation techniques for simulating the two-dimensional **Lebwohl–Lasher model**, which describes the orientational ordering of liquid crystals.

## Overview

The baseline Python implementation of the model was progressively optimised using a range of performance-enhancing methods. Each variant calculates the average orientational order parameter across different temperatures and grid sizes, allowing for a direct comparison of speed and scalability.

## Implementations

- **Core Python** – baseline serial version using nested loops  
- **NumPy** – introduces array operations for minor performance gains  
- **NumPy (Sequential)** – deterministic sequential updates for reproducibility  
- **NumPy (Vectorised)** – fully vectorised computation removing inner loops  
- **Numba** – JIT compilation using `@njit` for native-speed execution  
- **Numba (Parallel)** – multithreaded version with `@njit(parallel=True)`  
- **Cython** – compiled C-extension of the Python core  
- **Cython + OpenMP** – multithreaded Cython using OpenMP directives  
- **MPI** – distributed-memory parallelisation using `mpi4py`  
- **MPI + Numba / Cython** – hybrid approaches combining MPI with local acceleration  

## Results

All optimised versions produced identical physical results, confirming that the model’s integrity was maintained throughout optimisation.  
Runtime profiling showed substantial performance improvements, with **MPI + Cython** achieving the fastest execution overall.

## Purpose

This project demonstrates how Python, when combined with modern acceleration tools, can achieve near C-level performance.

