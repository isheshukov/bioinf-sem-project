# Отчёт номер 2

## OutOfAfrica без логарифмирования и со случайными начальными параметрами

```
>>> runfile('/home/ilia/src/projects/bioinf/sem2/project/sem-project/three_pop_bayes.py', wdir='/home/ilia/src/projects/bioinf/sem2/project/sem-project')
Beginning optimization ************************************************
MONKEY PATCHED BAYES START
WARNING:Inference:Model is < 0 where data is not masked.
WARNING:Inference:Number of affected entries is 735. Sum of data in those entries is 39.6288:
WARNING:Numerics:Extrapolation may have failed. Check resulting frequency spectrum for unexpected results.
WARNING:Inference:Model is < 0 where data is not masked.
WARNING:Inference:Number of affected entries is 4603. Sum of data in those entries is 9407.09:
num acquisition: 1, time elapsed: 1174.17s
WARNING:Numerics:Extrapolation may have failed. Check resulting frequency spectrum for unexpected results.
WARNING:Inference:Model is < 0 where data is not masked.
WARNING:Inference:Number of affected entries is 4689. Sum of data in those entries is 10726.4:
num acquisition: 2, time elapsed: 1980.81s
WARNING:Numerics:Extrapolation may have failed. Check resulting frequency spectrum for unexpected results.
num acquisition: 3, time elapsed: 2552.17s
WARNING:Inference:Model is < 0 where data is not masked.
WARNING:Inference:Number of affected entries is 4553. Sum of data in those entries is 3218.24:
WARNING:Inference:Model is < 0 where data is not masked.
num acquisition: 4, time elapsed: 2662.62s
WARNING:Inference:Number of affected entries is 1381. Sum of data in those entries is 145.873:
WARNING:Inference:Model is < 0 where data is not masked.
WARNING:Inference:Number of affected entries is 4620. Sum of data in those entries is 3923.43:
num acquisition: 5, time elapsed: 3397.29s
WARNING:Inference:Model is < 0 where data is not masked.
WARNING:Inference:Number of affected entries is 4646. Sum of data in those entries is 3297.95:
num acquisition: 6, time elapsed: 3805.89s
WARNING:Numerics:Extrapolation may have failed. Check resulting frequency spectrum for unexpected results.
WARNING:Inference:Model is < 0 where data is not masked.
WARNING:Inference:Number of affected entries is 4608. Sum of data in those entries is 10018.5:
num acquisition: 7, time elapsed: 5409.65s
WARNING:Inference:Model is < 0 where data is not masked.
WARNING:Inference:Number of affected entries is 3877. Sum of data in those entries is 1484.86:
num acquisition: 8, time elapsed: 5760.65s
WARNING:Numerics:Extrapolation may have failed. Check resulting frequency spectrum for unexpected results.
WARNING:Inference:Model is < 0 where data is not masked.
WARNING:Inference:Number of affected entries is 4681. Sum of data in those entries is 4727.22:
num acquisition: 9, time elapsed: 6101.27s
num acquisition: 10, time elapsed: 6455.11s
WARNING:Inference:Model is < 0 where data is not masked.
WARNING:Inference:Number of affected entries is 2925. Sum of data in those entries is 724.228:
[59.94494406 56.16254794 11.58362912 86.5325154  34.52535459 15.93475243
  2.30586125  0.71570458  8.27625146  1.06171279  2.30494396  0.33967538
  1.30043065]
MONKEY PATCHED BAYES END
Finshed optimization **************************************************
Maximum log composite likelihood: -25225.888178
WARNING:Inference:Model is < 0 where data is not masked.
Optimal value of theta: 210.386221036
WARNING:Inference:Number of affected entries is 3877. Sum of data in those entries is 1484.86:

```

Работало около часа. Результаты какие-то очень плохие, даже не близко к оптимуму.
