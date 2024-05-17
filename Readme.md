This repository contains our implementation of algorithms for PSI in the linear setting.
### Run the C++/ cython version 
The cpp files can be compiled and used as any `C++` library. Alternatively, there is a `cython3` binding that needs to be built in order to use EGE-SR/SH or APE in the benchmarks. 
To compile the cpp version with `cython` bindings follow the steps below: 

**Step 1** : Install the requirements with

`pip install -r requirements.txt`

**Step 2**: check the compiler settings in `setup.py` and run the following command from the base directory 

 `python3 setup.py build_ext --inplace`
 
**Step 3** import and use the algorithms as in the provided notebook file `test.ipynb`

### Reproduce experiments 
The notebook `test.ipynb` reproduces our results on the NoC dataset. 
 