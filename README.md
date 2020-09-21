This repository contains the code for :
* [Sobolev Training for Neural Networks](https://arxiv.org/abs/1706.04859)
* [Differential Machine Learning](https://arxiv.org/abs/2005.02347) 
* [Implicit Neural Representations with Periodic Activation Functions](https://arxiv.org/abs/2006.09661)

Test on different functions :
* Those used in Sobolev Training paper (see this [notebook](notebooks/optimization_functions.ipynb)) :
    * Ackley
    * Beale
    * Booth
    * Bukin
    * McCormick
    * Rosenbrock

* Those used in Differential Machine Learning paper (see this [notebook](notebooks/financial_functions.ipynb)) :
    * Pricing and Risk Functions : Black & Scholes
    * Gaussian Basket options : Bachelier dimension 1, 7, 20...

To summarize, you have the possibility to form a network following one of these processes :
* Normal Training (x, y) : with MLP and Siren
* Sobolev Training (x, y, dy/dx) : with MLP and Siren
* Twin_net tensorflow (x, y, dy/dx) : with MLP and Siren
* Twin_net pytorch (x, y, dy/dx) : with MLP and Siren

