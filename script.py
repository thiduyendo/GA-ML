# -*- coding: utf-8 -*-
"""
Created on Fri Apr 15 14:17:11 2022

@author: user
"""

import pandas as pd
import numpy as np
import csv, os
from numpy import savetxt
import random
#load train dataset
df=pd.read_csv('tcga_gbm_mgmt_status.csv')
X_trn= df.iloc[:,10:]
y_trnn = df['MGMT_status']
y_trn=y_trnn.copy()
y_trn=y_trn.replace({'UNMETHYLATED': 0},regex=True).replace({'METHYLATED': 1},regex=True)
X=X_trn.dropna(axis='columns')

#defining various steps required for the genetic algorithm
def initilization_of_population(size,n_feat):
    population = []
    for i in range(size):
        chromosome = np.ones(n_feat,dtype=np.bool)
        chromosome[:int(0.3*n_feat)]=False
        np.random.shuffle(chromosome)
        population.append(chromosome)
    return population

def fitness_score(population):
    scores = []
    newtp = []
    newfp = []
    newtn = []
    newfn = []
    #print('Population size: ', np.shape(population))
    for chromosome in population:
      tp = []
      fp = []
      tn = []
      fn = []
      acc = []
      for train, test in kfold.split(New_FS,y_trn): 
        # model = randomforest_model()   
        model.fit(New_FS.iloc[train].iloc[:,chromosome],y_trn[train])
        #evaluate the model
        true_labels = np.asarray(y_trn[test])
        predictions = model.predict(New_FS.iloc[test].iloc[:,chromosome])
        ntp, nfn, ntn, nfp= confusion_matrix(true_labels, predictions).ravel()
        tp.append(ntp)
        fp.append(nfp)
        tn.append(ntn)
        fn.append(nfn)
        acc.append(accuracy_score(true_labels, predictions))
      scores.append(np.mean(acc))
      newtp.append(np.sum(tp))
      newfp.append(np.sum(fp))
      newtn.append(np.sum(tn))
      newfn.append(np.sum(fn))
    scores, population = np.array(scores), np.array(population)
    
    weights=[]
    for score in scores:
      prob_score=score/(sum(scores))
      weights.append(prob_score)

    newtp, newfp, newtn, newfn = np.array(newtp), np.array(newfp), np.array(newtn), np.array(newfn)
    inds = np.argsort(scores)
    return list(scores[inds][::-1]), list(population[inds, :][::-1]) , list(np.array(weights)[inds][::-1]), list(newtp[inds][::-1]), list(newfp[inds][::-1]), list(newtn[inds][::-1]), list(newfn[inds][::-1])

def selection(pop_after_fit,weights,k):
    pop_after_sel = []
    selected_pop=random.choices(pop_after_fit, weights=weights, k=k)
    for t in selected_pop:
      pop_after_sel.append(t)
    # print('pop_after_sel:',pop_after_sel)
    # print(np.shape(pop_after_sel))
    return pop_after_sel

# crossover two parents to create two children
def crossover(p1, p2, crossover_rate):
	# children are copies of parents by default
	c1, c2 = p1.copy(), p2.copy()
	# check for recombination
	if random.random() < crossover_rate:
		# select crossover point that is not on the end of the string
		pt = random.randint(1, len(p1)-2)
		# perform crossover
		c1 = np.concatenate((p1[:pt],p2[pt:]))
		c2 = np.concatenate((p2[:pt],p1[pt:]))
	return [c1, c2]

def mutation(chromosome, mutation_rate):
	for i in range(len(chromosome)):
		# check for a mutation
		if random.random()  < mutation_rate:
			# flip the bit
			chromosome[i] = not chromosome[i]

def generations(size,n_feat,crossover_rate,mutation_rate, n_gen):
    best_chromo= []
    best_score= []
    population_nextgen=initilization_of_population(size,n_feat)
    max_score = []
    score = []
    for i in range(n_gen):
      scores, pop_after_fit,weights, tp, fp, tn, fn = fitness_score(population_nextgen)
      # print('score0',scores)
      score=scores[0]
      print('gen', i, score)
      k=size-2
      pop_after_sel = selection(pop_after_fit,weights,k) 
      # create the next generation
      children = list()
      for i in range(0, len(pop_after_sel), 2):
        # get selected parents in pairs
        p1, p2 = pop_after_sel[i], pop_after_sel[i+1]
        # crossover and mutation
        for c in crossover(p1, p2, crossover_rate):
          # mutation
          mutation(c, mutation_rate)
          # store for next generation
          children.append(c)
      # replace population
      pop_after_mutated = children
      population_nextgen=[]
      for c in pop_after_fit[:2]:
        population_nextgen.append(c)
      for p in pop_after_mutated:
        population_nextgen.append(p)

      best_chromo.append(pop_after_fit[0])
      best_score.append(score) 
    return best_chromo,best_score

#Running Genetic Algorithm
best_chromo,best_score=generations(size=50, n_feat=New_FS.shape[1],crossover_rate=0.8,mutation_rate=0.05,n_gen=5000)
