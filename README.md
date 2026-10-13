# ECSLR

[![License: GPL-3.0](https://img.shields.io/badge/License-GPL--3.0-blue.svg)](LICENSE)
[![Journal: EJOR](https://img.shields.io/badge/Journal-EJOR-informational.svg)](https://doi.org/10.1016/j.ejor.2025.07.028)

**ECSLR** is an R package implementing **Diverse Ensemble Cost-Sensitive Logistic Regression** for binary classification with asymmetric misclassification costs.

The method combines multiple cost-sensitive logistic regression models while retaining an interpretable logistic regression formulation. It uses sparsity and diversity regularization, with optimization based on partial conservative convex separable quadratic approximation (PCCSQA).

## Publication

Yang, B., Van Aelst, S., & Verdonck, T. (2026). **Diverse ensemble cost-sensitive logistic regression.** *European Journal of Operational Research*, **328**(1), 282–294.

**Paper:** https://doi.org/10.1016/j.ejor.2025.07.028

[Read the article on ScienceDirect](https://www.sciencedirect.com/science/article/pii/S0377221725005545)

## Installation

Install the development version from GitHub:

```r
install.packages("remotes")
remotes::install_github("bingStat/ECSLR")
```

The package includes compiled C++ code and requires an appropriate R compilation toolchain (such as Rtools on Windows). Required R dependencies are specified in [DESCRIPTION](DESCRIPTION).

## Quick start

```r
library(ECSLR)

set.seed(2026)
n <- 100
x <- matrix(rnorm(n * 5), nrow = n)
y <- rbinom(n, size = 1, prob = plogis(0.8 * x[, 1] - 0.5 * x[, 2]))

# Columns: true negative, false positive, false negative, true positive
cost_matrix <- cbind(
  TN = rep(0, n),
  FP = rep(1, n),
  FN = rep(5, n),
  TP = rep(0, n)
)

fit <- ECSLR(
  x, y, cost_matrix,
  M = 5,
  lambda_sparsity = 0.01,
  lambda_diversity = 0.01
)

coef(fit)

pred <- predict(
  fit,
  newx = x,
  new_costmatrix = cost_matrix,
  type = "class",
  threshold.criteria = "ID-cost"
)
table(predicted = pred, observed = y)
```

The cost matrix has **one row per observation** and four columns, ordered **TN, FP, FN, TP**. Costs can vary between observations.

For full function signatures and options, see the package help pages: `?ECSLR`, `?cv.ECSLR`, and `?predict.ECSLR`.

## Citation

If you use ECSLR in your research, please cite:

```bibtex
@article{yang2026diverse,
  title   = {Diverse ensemble cost-sensitive logistic regression},
  author  = {Yang, Bing and Van Aelst, Stefan and Verdonck, Tim},
  journal = {European Journal of Operational Research},
  volume  = {328},
  number  = {1},
  pages   = {282--294},
  year    = {2026},
  doi     = {10.1016/j.ejor.2025.07.028}
}
```

## License

Distributed under the [GNU General Public License version 3](LICENSE), consistent with `License: GPL-3` in the R package metadata.
