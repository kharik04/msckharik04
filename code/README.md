# SBB challenge ADMM solver and MILP paser

In this folder you will find the code developed for the thesis Mixed Integer Linear Programming for the Train Timetabling Problem. In order to understand the code and its intent, please refer to the thesis.

The code consists of two scipts; SBB2MILP.py and solver.py. There is also a folder with SBB's test cases attached.

SBB2MILP.py contains the parser itself, returning constraint matrices and other various data structures needed to run the solver. 

solver.py is the main file. It contains both the ADMM implementation as well as naive solver.


NB: The code is developed as proof of concept, and although time and space complexities were taken into consideration, it is by no means intended to be commercial grade software due to choice of programming language as well as implementation design. 

## Dependencies

-scipy  
-numpy  
-networkx  
-mosek  
-seaborn  
-matplotlib
-valid mosek license. During this thesis, a trial license was used, but MOSEK offers academic liscenses.


## Usage
```console
:~$ python3 solver.py algorithm=ALGORITHM display=DISPLAY file=FILE
```

The options are: algorithm = [naive, ADMM], display = [True, False] and file is the location to the json file of a SBB scenario. For example to run ADMM without displaying on test case 01 we call

```console
:~$ python3 solver.py algorithm=ADMM display=False file=problem_instances/01_dummy.json
```