# Predicting Goldbach Partitions
Goldbach's conjecture is one of the oldest and best-known unsolved problems in number theory and all of mathematics.
It states:
Every even number greater than 2 can be expressed as the sum of 2 prime numbers.
The conjecture has been shown to hold for all integers less than 4*10^18 but remains unproven despite considerable effort.

We would like to build a model that would be a good approximation of the Goldbach function. Where the Goldbach function is defined as:
g(E) is defined for all even integers E>2 to be the number of different ways, 
in which E can be expressed as the sum of two primes.
For Example g(6) = 1 since 6 can be expressed as the sum of 2 primes in 1 way
6=3+3
## Data Set
Our work horse data set contains 32k numbers (from 4 to 64k) and the number of their partitions.
## Linear Regression
With this model we achived MSE of ~5.7k
## Multi Linear Preceptron
With this model we achived MSE of ~600
## Long Short Term memory
With this model we Achived MSE of 100k


## Authors

* **Ehud Plaksin** - *Initial work* - 


