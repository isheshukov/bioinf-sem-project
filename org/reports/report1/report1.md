Что нужно было сделать:

* [x] В мануале dadi или moments найти модель для трех популяций YRI, CEU, CHB и запустить байеса на ней.
* [x] Попробовать разные Acquisiton functions, параметрические модели
* [x] Научиться запускаться с ограничением по времени и сравнить likelihoods
* [x] Графики сходимости


# Запуск с ограничением по времени. Графики. Acquisition functions

## ∂a∂i
Dataset: YRI_CEU

| Время (sec)    | Log likelihood | Theta       | Plot
|---------------:|---------------:|------------:|:----------------------------|
| 5              |-1086.74431567  |2857.57702173| ![img](dadi_YRI_CEU_5s.png) |
| 10             |-1132.99393383  |2836.68844085| ![img](dadi_YRI_CEU_10s.png)|
| 30             |-1068.56729965  |2757.73218775| ![img](dadi_YRI_CEU_30s.png)|
| 60             |-1066.26773378  |2746.74165041| ![img](dadi_YRI_CEU_60s.png)|

## GPyOpt
Dataset: YRI_CEU

У модуля bayesian_optimization есть несколько разных acquisition functions:
	
`‘EI’, expected improvement. - ‘EI_MCMC’, integrated expected improvement (requires GP_MCMC model). - ‘MPI’, maximum probability of improvement. - ‘MPI_MCMC’, maximum probability of improvement (requires GP_MCMC model). - ‘LCB’, GP-Lower confidence bound. - ‘LCB_MCMC’, integrated GP-Lower confidence bound (requires GP_MCMC model).`

Я остановлюсь на EI, MPI и LCB.
За параметр `max_iter` взято число `100`.
Время -- параметр `max_time`.

```
max_iter – exploration horizon, or number of acquisitions. If nothing is provided optimizes the current acquisition.
max_time – maximum exploration horizon in seconds.
```

# EI

| Время (sec)    | Log likelihood | Theta       | Plot (dadi)                   | Convergence plot              |
|---------------:|---------------:|------------:|:-----------------------------:|:-----------------------------:|
| 5              |-1413.33998204  |2619.6841977 | ![img](gpy_YRI_CEU_5s_EI.png) | ![img](gpy_YRI_CEU_5s_EI_conv.png) |
| 10             |-1474.55524648  |3174.99740624| ![img](gpy_YRI_CEU_10s_EI.png)| ![img](gpy_YRI_CEU_10s_EI_conv.png)
| 30             |-1702.12160555  |2694.28253087| ![img](gpy_YRI_CEU_30s_EI.png)| ![img](gpy_YRI_CEU_30s_EI_conv.png)
| 60             |-1331.0171545   |2799.10106502| ![img](gpy_YRI_CEU_60s_EI.png)| ![img](gpy_YRI_CEU_60s_EI_conv.png)

# MPI

| Время (sec)    | Log likelihood | Theta       | Plot (dadi)                   | Convergence plot                      |
|---------------:|---------------:|------------:|:-----------------------------:|:-------------------------------------:|
| 5              |-1628.91303751  |2940.22119622| ![img](gpy_YRI_CEU_5s_MPI.png) | ![img](gpy_YRI_CEU_5s_MPI_conv.png)  |
| 10             |-1809.43657418  |2381.30972935| ![img](gpy_YRI_CEU_10s_MPI.png)| ![img](gpy_YRI_CEU_10s_MPI_conv.png)
| 30             |-1415.61997519  |2542.22977459| ![img](gpy_YRI_CEU_30s_MPI.png)| ![img](gpy_YRI_CEU_30s_MPI_conv.png)

# LCB

| Время (sec)    | Log likelihood | Theta       | Plot (dadi)                   | Convergence plot                      |
|---------------:|---------------:|------------:|:-----------------------------:|:-------------------------------------:|
| 5              |-1658.62452068  |3374.40224855| ![img](gpy_YRI_CEU_5s_LCB.png) | ![img](gpy_YRI_CEU_5s_LCB_conv.png)  |
| 10             |-1284.20224929  |2441.54310219| ![img](gpy_YRI_CEU_10s_LCB.png)| ![img](gpy_YRI_CEU_10s_LCB_conv.png)
| 30             |-1400.49740334  |2372.55734299| ![img](gpy_YRI_CEU_30s_LCB.png)| ![img](gpy_YRI_CEU_30s_LCB_conv.png)

# YRI CEU CHB

| ∂a∂i (from manual)                               | GPyOpt (EI)                                      |
|--------------------------------------------------|--------------------------------------------------|
| ![img](dadi-1d_comp-1.png)                       | ![img](gpyopt_1d_comp.png)                       |
| ![img](dadi-2d_comp-1.png)                       | ![img](gpyopt_2d_comp.png)                       |
| ![img](dadi-2d_single-1.png)                     | ![img](gpyopt_2d_single.png)                     |
| ![img](dadi-3d_comp-1.png)                       | ![img](gpyopt_3d_comp.png)                       |
| Maximum log composite likelihood: -10437.1839666 | Maximum log composite likelihood: -6316.96272061 |
| Optimal value of theta: 2715.66860633            | Optimal value of theta: 2761.19105232            |