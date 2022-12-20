# -*- coding:utf-8 -*-

import numpy as np
import pandas as pd
from functools import reduce

## Function to create a conditional probability table
## Conditional probability is of the form p(x1 | x2, ..., xk)
## varnames: vector of variable names (strings) first variable listed 
##           will be x_i, remainder will be parents of x_i, p1, ..., pk
## probs: vector of probabilities for the flattened probability table
## outcomesList: a list containing a vector of outcomes for each variable
## factorTable is in the type of pandas dataframe
## See the test file for examples of how this function works
def readFactorTable(varnames, probs, outcomesList):
    factorTable = pd.DataFrame({'probs': probs})

    totalfactorTableLength = len(probs)
    numVars = len(varnames)

    k = 1
    for i in range(numVars - 1, -1, -1):
        levs = outcomesList[i]
        numLevs = len(levs)
        col = []
        for j in range(0, numLevs):
            col = col + [levs[j]] * k
        factorTable[varnames[i]] = col * int(totalfactorTableLength / (k * numLevs))
        k = k * numLevs

    return factorTable

## Build a factorTable from a data frame using frequencies
## from a data frame of data to generate the probabilities.
## data: data frame read using pandas read_csv
## varnames: specify what variables you want to read from the table
## factorTable is in the type of pandas dataframe
def readFactorTablefromData(data, varnames):
    numVars = len(varnames)
    outcomesList = []

    for i in range(0, numVars):
        name = varnames[i]
        outcomesList = outcomesList + [list(set(data[name]))]

    lengths = list(map(lambda x: len(x), outcomesList))
    m = reduce(lambda x, y: x * y, lengths)
   
    factorTable = pd.DataFrame({'probs': np.zeros(m)})

    k = 1
    for i in range(numVars - 1, -1, -1):
        levs = outcomesList[i]
        numLevs = len(levs)
        col = []
        for j in range(0, numLevs):
            col = col + [levs[j]] * k
        factorTable[varnames[i]] = col * int(m / (k * numLevs))
        k = k * numLevs

    numLevels = len(outcomesList[0])

    # creates the vector called fact to index probabilities 
    # using matrix multiplication with the data frame
    fact = np.zeros(data.shape[1])
    lastfact = 1
    for i in range(len(varnames) - 1, -1, -1):
        fact = np.where(np.isin(list(data), varnames[i]), lastfact, fact)
        lastfact = lastfact * len(outcomesList[i])

    # Compute unnormalized counts of subjects that satisfy all conditions
    a = (data - 1).dot(fact) + 1
    for i in range(0, m):
        factorTable.at[i,'probs'] = sum(a == (i+1))

    # normalize the conditional probabilities
    skip = int(m / numLevels)
    for i in range(0, skip):
        normalizeZ = 0
        for j in range(i, m, skip):
            normalizeZ = normalizeZ + factorTable['probs'][j]
        for j in range(i, m, skip):
            if normalizeZ != 0:
                factorTable.at[j,'probs'] = factorTable['probs'][j] / normalizeZ

    return factorTable


## Join of two factors
## Factor1, Factor2: two factor tables
##
## Should return a factor table that is the join of factor 1 and 2.
## You can assume that the join of two factors is a valid operation.
## Hint: You can look up pd.merge for mergin two factors
def joinFactors(Factor1, Factor2):
    # your code 
    Factor1 = Factor1.rename(columns={"probs": "probs1"})
    Factor2 = Factor2.rename(columns={"probs": "probs2"})
    if Factor1.columns.intersection(Factor2.columns).empty:
        # no common variables, calculate Cartesian product
        how = "cross"
    else:
        # join on common variables
        how = "inner"
    res = pd.merge(Factor1, Factor2, how=how)
    # calculate the joint probability
    res["probs"] = res["probs1"] * res["probs2"]
    res = res.drop(labels=["probs1", "probs2"], axis=1)
    return res

## Marginalize a variable from a factor
## table: a factor table in dataframe
## hiddenVar: a string of the hidden variable name to be marginalized
##
## Should return a factor table that marginalizes margVar out of it.
## Assume that hiddenVar is on the left side of the conditional.
## Hint: you can look can pd.groupby
def marginalizeFactor(factorTable, hiddenVar):
    # your code 
    res = factorTable.copy()
    if hiddenVar in res.columns:
        # group by variables except hiddenVar
        by = res.columns.drop([hiddenVar, "probs"])
        grouped = res.groupby(list(by))
        # sum over hiddenVar in each group
        res["probs"] = grouped['probs'].transform('sum')
        res.drop(labels=[hiddenVar], axis=1, inplace=True)
        res.drop_duplicates(inplace=True)
    return res

## Update BayesNet for a set of evidence variables
## bayesnet: a list of factor and factor tables in dataframe format
## evidenceVars: a list of variable names in the evidence list
## evidenceVals: a list of values for corresponding variables (in the same order)
##
## Set the values of the evidence variables. Other values for the variables
## should be removed from the tables. You do not need to normalize the factors
def evidenceUpdateNet(bayesnet, evidenceVars, evidenceVals):
    # your code 
    res = []
    for df in bayesnet:
        df_res = df.copy()
        for i in range(len(evidenceVars)):
            var = evidenceVars[i]
            val = evidenceVals[i]
            if var in df_res.columns:
                # delete rows where var != val
                df_res.drop(df_res[df_res[var] != val].index, inplace=True)
        res.append(df_res)
    return res


## Run inference on a Bayesian network
## bayesnet: a list of factor tables and each table iin dataframe type
## hiddenVars: a list of variable names to be marginalized
## evidenceVars: a list of variable names in the evidence list
## evidenceVals: a list of values for corresponding variables (in the same order)
##
## This function should run variable elimination algorithm by using 
## join and marginalization of the sets of variables. 
## The order of the elimiation can follow hiddenVar ordering
## It should return a single joint probability table. The
## variables that are hidden should not appear in the table. The variables
## that are evidence variable should appear in the table, but only with the single
## evidence value. The variables that are not marginalized or evidence should
## appear in the table with all of their possible values. The probabilities
## should be normalized to sum to one.
def inference(bayesnet, hiddenVars, evidenceVars, evidenceVals):
    # your code 
    # update net by evidences
    net = evidenceUpdateNet(bayesnet, evidenceVars, evidenceVals)

    # join and eliminate all hidden variables
    for oneHiddenVar in hiddenVars:
        # join all factors containing `oneHiddenVar`
        jointFactor = pd.DataFrame()
        for i in range(len(net)-1, -1, -1):
            if oneHiddenVar in net[i].columns:
                if jointFactor.empty:
                    jointFactor = net[i]
                else:
                    jointFactor = joinFactors(jointFactor, net[i])
                # remove `net[i]` from `net`
                del net[i]
        if not jointFactor.empty:
            # eliminate `oneHiddenVar`
            jointFactor = marginalizeFactor(jointFactor, oneHiddenVar)
            # add the resultant factor to the net
            net.append(jointFactor)

    # join all remaining factors
    res = net[0]
    for factor in net[1:]:
        res = joinFactors(res, factor)

    # normalize
    res["probs"] /= res["probs"].sum()

    return res


## you can add other functions as you wish.
def my_function():
    return
