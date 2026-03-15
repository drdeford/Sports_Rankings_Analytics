# -*- coding: utf-8 -*-
"""
Created on Sun Mar 15 16:34:48 2026

@author: eungm
"""

import math
import random
import numpy as np
from scipy.stats import beta
import matplotlib.pyplot as plt

class Player:
  G: float
  S: float
  R: float
  eps: float
  M: list

  def __init__(self, pid):
      self.pid = pid
      # G_list is a nested list containing every possible growth factor
      G_list=[]
      for i in range(5):
        G_list.append([])
        for j in range(5):
          G_list[i].append(0.96+((i+j)/100))

      G = random.choice(G_list)
      # code S generation to take from skewed right distribution
      # beta distribution, median = 3.5
      S = beta.rvs(2, 5)*10
      eps = 1.5-(0.5*((math.log(S+1, math.e))/(math.log(11, math.e))))
      # code initial rating to round to nearest tenth
      R = round((eps*S), 1)

  def new_eps(playerE, playerS):
    playerE = 1.5-(0.5*((math.log(playerS+1, math.e))/(math.log(11, math.e))))
    return playerE

  def new_R(playerR, playerS, playerE):
    past_r = playerR
    playerR = round(playerE*playerS, 1)

    if playerR-past_r >= 0:
      print(f"+{playerR-past_r}")
    else:
      print(f"-{playerR - past_r}")
    return playerR

  def update(playerG, playerS, playerE, playerR):
    currentG = random.choice(playerG)
    playerS = currentG*playerS
    playerE = Player.new_eps(playerE, playerS)
    playerR = Player.new_R(playerR, playerS, playerE)
    return playerS, playerE, playerR

