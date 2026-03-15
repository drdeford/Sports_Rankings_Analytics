# -*- coding: utf-8 -*-
"""
Created on Fri Mar 13 17:02:30 2026

@author: eungm
"""

# %% import

from __future__ import annotations

from Soccer_Simulation_Helpers import *

import pandas as pd

import random

# %% for loop

for c in range(6):
  results = []
  for s in range(1000):
    if c < 6:
      sampler, baselines = tiered_conference_baselines(
          n_conferences=2, n_strong=0, strong_mean=2, weak_mean=0, within_sigma=1
      )
      strong = 0
      if c == 3:
        sampler, baselines = tiered_conference_baselines(
            n_conferences=2, n_strong=1, strong_mean=2, weak_mean=0, within_sigma=1
        )
        strong = 1

    if c < 5:
      teams, games = build_schedule_with_conferences_mixed_nonconf(
          n_conferences=2,
          teams_per_conference=8,
          k_nonconf=0,
          strength_sampler=sampler,
          balance_across_conferences=True
      )
      k = 0
      if c < 4:
          teams, games = build_schedule_with_conferences_mixed_nonconf(
              n_conferences=2,
              teams_per_conference=8,
              k_nonconf=1,
              strength_sampler=sampler,
              balance_across_conferences=True
          )
          k = 1
          if c < 2:
              teams, games = build_schedule_with_conferences_mixed_nonconf(
                  n_conferences=2,
                  teams_per_conference=8,
                  k_nonconf=4,
                  strength_sampler=sampler,
                  balance_across_conferences=True
              )
              k = 4
    else:
      teams, games = build_schedule_with_conferences_mixed_nonconf(
          n_conferences=2,
          teams_per_conference=8,
          k_nonconf=4,
          strength_sampler=sampler,
          balance_across_conferences=True
      )
      k = 4
    if c < 2:
      teams[0] = Team(tid = 0, conference = 0, strength = 2.5)
      if c < 1:
        teams[1] = Team(tid = 0, conference = 0, strength = 2.5)

    gdf = simulate_results(games=games, teams=teams, stochastic= True)

    npi_df = calculate_npi(gdf)

    strength = [x.strength for x in teams]
    npi = [0 for x in teams]
    wins = [0 for x in teams]
    losses = [0 for x in teams]
    conf = [x.conference for x in teams]

    for index, row in npi_df.iterrows():

        npi[int(row['team'])] = row['npi']

    for index, row in gdf.iterrows():

        if row['home_score'] ==1:
            wins[row['home_team']] += 1
            losses[row['away_team']] += 1
        else:
            wins[row['away_team']] += 1
            losses[row['home_team']] += 1

    npi_calibration_report(strength, npi, wins, losses, conf, topk_list=(2,4,8))
    if c < 2:
      confcol_one = pd.Series(npi[1:8])
      if c < 1:
        confcol_one = pd.Series(npi[2:8])
    else:
      confcol_one = pd.Series(npi[0:8])
    confcol_two = pd.Series(npi[8:15])
    mean_diff = confcol_one.mean() - confcol_two.mean()
    results.append(mean_diff)

  season_one = pd.Series(results)

  print(c)
  print(season_one.mean())

 # print(teams[0].strength, teams[1].strength, k, strong)
 # print(confcol_one)
#season_one = pd.DataFrame(results, index = ["first", "second", "third", "fourth", "fifth", "sixth", "seventh", "eighth", "ninth", "tenth"])

#print(season_one)

# %% graphs

plt.scatter(strength,npi,c=conf,cmap='jet')
plt.xlabel('Strength')
plt.ylabel('NPI')
plt.show()

plt.scatter(wins,npi,c=conf,cmap='jet')
plt.xlabel('Wins')
plt.ylabel('NPI')
plt.show()

cws_net = nx.Graph()

for row in games:
    cws_net.add_edge(row[0],row[1])

for node in range(len(teams)):

    cws_net.nodes()[node]['CONF'] = conf[node]
    cws_net.nodes()[node]['WINS'] = wins[node]
    cws_net.nodes()[node]['NPI'] = npi[node]
    cws_net.nodes()[node]['S'] = strength[node]

kk_layout = nx.kamada_kawai_layout(cws_net)
nx.draw(cws_net,node_color=[cws_net.nodes()[node]['NPI'] for node in cws_net.nodes()],cmap='jet')
plt.show()
nx.draw(cws_net,node_color=[cws_net.nodes()[node]['NPI'] for node in cws_net.nodes()],node_size=[10*(10+cws_net.nodes()[node]['S']) for node in cws_net.nodes()],cmap='jet')
plt.show()
nx.draw(cws_net,node_color=[cws_net.nodes()[node]['CONF'] for node in cws_net.nodes()],cmap='jet')
plt.show()