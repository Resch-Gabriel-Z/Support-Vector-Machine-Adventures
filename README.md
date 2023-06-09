# My Support Vector Machine Adventures

# Introduction

In this repository, I try to implement various SVM's and demonstrate their results.
My main goal is to learn a fundamental and important concept of Machine Learning as well as to provide clean code
for other people to learn from it (but I think it's important to point out, that I'm not a teacher and there are
various other and better resources available).

# Types of SVMs

## Linear SVMs

This kind of SVM focuses on creating a linear hyperplane by maximizing a margin between 2 linear separable
classes.

## Multi-class SVM
A type of SVM that works on classifying a dataset between more than 2 classes. Usually this is achieved by training
several binary SVMs and build on top of that a separation.

## Kernel SVMs

or non-linear SVMs, they use Kernel methods to map a Dataset, which is non-linear
to a higher dimension N, for which a Hyperplane exists.

# Installation & Usage

Feel free to download this repository and change it however you want, I didn't try to make a software out of it that
requires special UI, or something like that.

Its really just a bunch of scripts in Python. If you just want to Train/Test out the Algorithms for yourself, you can
easily do this by:

1. Clone the repository to your local machine:

```bash
git clone https://github.com/Resch-Gabriel-Z/Support-Vector-Machine-Adventures
```

2. Change the meta-data to appropriate Data
3. Run the Training Script

# Results

## Linear SVM

### Hard-margin
![HM1.png](media%2FHM1.png)
![HM2.png](media%2FHM2.png)
![HM3.png](media%2FHM3.png)
![HM4.png](media%2FHM4.png)
### Soft-margin
![SM1.png](media%2FSM1.png)
![SM2.png](media%2FSM2.png)
![SM3.png](media%2FSM3.png)

## Multi-Class SVM

### OAA
![OAA1.png](media%2FOAA1.png)
![OAA2.png](media%2FOAA2.png)
![OAA3.png](media%2FOAA3.png)
![OAA4.png](media%2FOAA4.png)