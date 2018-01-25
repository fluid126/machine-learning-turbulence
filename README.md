# Machine Learning in Turbulence Modeling

This project aims to utilize a machine learning method proposed by Ling et al., the tensor basis neural network (TBNN), to learn a model for the Reynolds stress anisotropy tensor of a turbulent channel flow from the DNS data. The data used in the project is obtained from the Johns Hopkins Turbulence Database (JHTDB). See full status report [here](https://github.com/fr0420/machine-learning-turbulence/blob/master/project_report_v0.pdf).

## Requirements 
To run the codes in this repository, make sure you have installed the following packages: 
1. pyJHTDB at [https://github.com/idies/pyJHTDB](https://github.com/idies/pyJHTDB)  
2. tbnn at [https://github.com/tbnn/tbnn](https://github.com/tbnn/tbnn) 

Notice that different python versions are required for codes in different folders:  
- ./JHTDB -> Python 3 
- ./TBNN -> Python 2 


