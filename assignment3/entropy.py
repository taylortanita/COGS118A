import numpy as np
import math

def entropy(values, num):
  rtn_val = 0
  for i in range(num):
    rtn_val = rtn_val + values[i]*math.log(values[i])
  return rtn_val*(-1)

def conditional_entropy(values, num_rows, num_col):
  rtn = 0
  for j in range(num_col):
    denominator = 0
    
    for i in range(num_rows):
      denominator = denominator + values[i,j]
    
    ent = 0
    for i in range(num_rows):
      x = values[i,j]/denominator
      ent = ent + x*math.log(x)
  
    rtn = rtn + denominator*ent
  return rtn*(-1)

def mutual_entropy(values, num_rows, num_col):
  rtn = 0
  x_sum = np.sum(values, axis=0)
  y_sum = np.sum(values, axis=1)

  for j in range(num_col):
    for i in range(num_rows):
      rtn = rtn + values[i,j]*math.log(values[i,j]/(x_sum[i]*y_sum[j]))
  return rtn

