# -*- coding:utf-8 -*-

from BayesianNetworks import *
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#############################
## Example Tests from Bishop `Pattern Recognition and Machine Learning` textbook on page 377
#############################
BatteryState = readFactorTable(['battery'], [0.9, 0.1], [[1, 0]])
FuelState = readFactorTable(['fuel'], [0.9, 0.1], [[1, 0]])
GaugeBF = readFactorTable(['gauge', 'battery', 'fuel'], [0.8, 0.2, 0.2, 0.1, 0.2, 0.8, 0.8, 0.9], [[1, 0], [1, 0], [1, 0]])

carNet = [BatteryState, FuelState, GaugeBF] # carNet is a list of factors 
## Notice that different order of operations give the same answer
## (rows/columns may be permuted)
joinFactors(joinFactors(BatteryState, FuelState), GaugeBF)
joinFactors(joinFactors(GaugeBF, FuelState), BatteryState)

marginalizeFactor(joinFactors(GaugeBF, BatteryState), 'gauge')
joinFactors(marginalizeFactor(GaugeBF, 'gauge'), BatteryState)

joinFactors(marginalizeFactor(joinFactors(GaugeBF, BatteryState), 'battery'), FuelState)
marginalizeFactor(joinFactors(joinFactors(GaugeBF, FuelState), BatteryState), 'battery')

marginalizeFactor(joinFactors(marginalizeFactor(joinFactors(GaugeBF, BatteryState), 'battery'), FuelState), 'gauge')
marginalizeFactor(joinFactors(marginalizeFactor(joinFactors(GaugeBF, BatteryState), 'battery'), FuelState), 'fuel')

evidenceUpdateNet(carNet, ['fuel', 'battery'], [1, 0])

# inference
print("inference starts")
print(inference(carNet, ['battery', 'fuel'], [], []) )        ## chapter 8 equation (8.30)
print(inference(carNet, ['battery'], ['fuel'], [0]))           ## chapter 8 equation (8.31)
print(inference(carNet, ['battery'], ['gauge'], [0]))          ##chapter 8 equation  (8.32)
print(inference(carNet, [], ['gauge', 'battery'], [0, 0]))    ## chapter 8 equation (8.33)
print("inference ends")
###########################################################################
#RiskFactor Data Tests
###########################################################################
riskFactorNet = pd.read_csv('RiskFactorsData.csv')

# Create factors

income      = readFactorTablefromData(riskFactorNet, ['income'])
smoke       = readFactorTablefromData(riskFactorNet, ['smoke', 'income'])
exercise    = readFactorTablefromData(riskFactorNet, ['exercise', 'income'])
long_sit    = readFactorTablefromData(riskFactorNet, ['long_sit', 'income'])
stay_up     = readFactorTablefromData(riskFactorNet, ['stay_up', 'income'])
bmi         = readFactorTablefromData(riskFactorNet, ['bmi', 'income'])
diabetes    = readFactorTablefromData(riskFactorNet, ['diabetes', 'bmi'])

## you need to create more factor tables

risk_net = [income, smoke, long_sit, stay_up, exercise, bmi, diabetes]
print("income dataframe is ")
print(income)
factors = riskFactorNet.columns

# example test p(diabetes|smoke=1,exercise=2,long_sit=1)

margVars = list(set(factors) - {'diabetes', 'smoke', 'exercise','long_sit'})
obsVars  = ['smoke', 'exercise','long_sit']
obsVals  = [1, 2, 1]

p = inference(risk_net, margVars, obsVars, obsVals)
print(p)

 
###########################################################################
# Please write your own test script
# HW3 test scripts start from here
###########################################################################
print("--------------------------------------------------------------------")
print("Below are program results of the written part.\n")

# Problem 1
print("####################################################")
print("Problem 1\n")
# build all factors
income      = readFactorTablefromData(riskFactorNet, ['income'])
smoke       = readFactorTablefromData(riskFactorNet, ['smoke', 'income'])
exercise    = readFactorTablefromData(riskFactorNet, ['exercise', 'income'])
long_sit    = readFactorTablefromData(riskFactorNet, ['long_sit', 'income'])
stay_up     = readFactorTablefromData(riskFactorNet, ['stay_up', 'income'])
bmi         = readFactorTablefromData(riskFactorNet, ['bmi', 'income', 'exercise', 'long_sit'])
bp          = readFactorTablefromData(riskFactorNet, ['bp', 'income', 'exercise', 'long_sit', 'stay_up', 'smoke'])
cholesterol = readFactorTablefromData(riskFactorNet, ['cholesterol', 'income', 'exercise', 'stay_up', 'smoke'])
diabetes    = readFactorTablefromData(riskFactorNet, ['diabetes', 'bmi'])
stroke      = readFactorTablefromData(riskFactorNet, ['stroke', 'bmi', 'bp', 'cholesterol'])
attack      = readFactorTablefromData(riskFactorNet, ['attack', 'bmi', 'bp', 'cholesterol'])
angina      = readFactorTablefromData(riskFactorNet, ['angina', 'bmi', 'bp', 'cholesterol'])
# build the network
whole_net = [income,
            smoke, exercise, long_sit, stay_up,
            bp, cholesterol, bmi,
            diabetes, stroke, attack, angina]
# infer in this complete network
whole_p = inference(whole_net, [], [], [])
print(f"The size of this network is {len(whole_p)}.")
print()


# Problem 2
print("####################################################")
print("Problem 2\n")
health_outcomes = ['diabetes', 'stroke', 'attack', 'angina']
habits = ['smoke', 'exercise', 'long_sit', 'stay_up']
health = ['bp', 'cholesterol', 'bmi']

def analyse_habit_health(net, prt=1, plot=1):
    """
    Analyse effects of habits and health on health outcomes and
    return a dictionary storing P(outcome = 1 | evidence) for each outcome and evidence.
    The parameter `prt` controls whether to print results and
    `plot` controls whether to plot results.
    """
    probs = {}
    for outcome in health_outcomes:
        outcome_p = pd.DataFrame()
        # (a)
        hiddenVars = list(set(riskFactorNet.columns) - {outcome} - set(habits))
        # bad habits
        bad_habit_p = inference(net, hiddenVars, habits, [1, 2, 1, 1])
        # print(f"If I have bad habits, then the probabilities of {outcome} are:")
        # print(bad_habit_p[[outcome, 'probs']])
        outcome_p[outcome] = bad_habit_p[outcome]
        outcome_p['bad habits'] = bad_habit_p['probs']
        # good habits
        good_habit_p = inference(net, hiddenVars, habits, [2, 1, 2, 2])
        # print(f"If I have good habits, then the probabilities of {outcome} are:")
        # print(good_habit_p[[outcome, 'probs']])
        outcome_p['good habits'] = good_habit_p['probs']

        # (b)
        hiddenVars = list(set(riskFactorNet.columns) - {outcome} - set(health))
        # poor health
        poor_health_p = inference(net, hiddenVars, health, [1, 1, 3])
        # print(f"If I have poor health, then the probabilities of {outcome} are:")
        # print(poor_health_p[[outcome, 'probs']])
        outcome_p['poor health'] = poor_health_p['probs']
        # good health
        good_health_p = inference(net, hiddenVars, health, [3, 2, 2])
        # print(f"If I have good health, then the probabilities of {outcome} are:")
        # print(good_health_p[[outcome, 'probs']])
        outcome_p['good health'] = good_health_p['probs']

        # record P(outcome=1 | evidence)
        # probs[outcome] is a series containing probabilities of the outcome with the four evidence
        probs[outcome] = outcome_p[outcome_p[outcome]==1][
            ['bad habits', 'good habits', 'poor health', 'good health']].iloc[0]

        if prt:
            print(f"Probabilities of {outcome} given conditions of habits or health:")
            print(outcome_p)
            print()
        
        if plot:
            evi_pos = np.arange(4)  # positions of labels of evidences
            plt.bar(evi_pos, probs[outcome], width=0.3)
            plt.xticks(evi_pos, labels=['bad habits', 'good habits', 'poor health', 'good health'])
            plt.xlabel(f'condition')
            plt.ylabel(f'probability of {outcome}')
            plt.show()
    
    return probs

p_without_habit = analyse_habit_health(whole_net)
print()


# Problem 3
print("####################################################")
print("Problem 3\n")
for outcome in health_outcomes:
    income_outcome_p = pd.DataFrame()
    income_outcome_p['income'] = np.arange(1, len(income) + 1)
    probs = []
    hiddenVars = list(set(riskFactorNet.columns) - {'income', outcome})
    for income_level in range(1, len(income)+1):
        income_level_p = inference(whole_net, hiddenVars, ['income'], [income_level])
        probs.append(income_level_p[income_level_p[outcome]==1].iloc[0]['probs'])
    income_outcome_p['probs'] = probs
    # output P(outcome=1 | income) for different incomes
    print(f"Probability of {outcome} given different incomes:")
    print(income_outcome_p)
    print()
    income_level = list(range(1, len(income)+1))
    outcome_prob = income_outcome_p['probs']
    plt.plot(income_level, outcome_prob)
plt.legend(health_outcomes)
plt.xlabel('level of income')
plt.ylabel('probability of outcome')
plt.show()
print()


# Problem 4
print("####################################################")
print("Problem 4\n")
diabetes_p4 = readFactorTablefromData(riskFactorNet, ['diabetes', 'bmi', 'smoke', 'exercise'])
stroke_p4   = readFactorTablefromData(riskFactorNet, ['stroke', 'bmi', 'bp', 'cholesterol', 'smoke', 'exercise'])
attack_p4   = readFactorTablefromData(riskFactorNet, ['attack', 'bmi', 'bp', 'cholesterol', 'smoke', 'exercise'])
angina_p4   = readFactorTablefromData(riskFactorNet, ['angina', 'bmi', 'bp', 'cholesterol', 'smoke', 'exercise'])
whole_net_p4 = [income,
            smoke, exercise, long_sit, stay_up,
            bp, cholesterol, bmi,
            diabetes_p4, stroke_p4, attack_p4, angina_p4]

p_with_habit = analyse_habit_health(whole_net_p4, plot=0)

# plot the bar plot to compare
evi_pos = np.arange(4)  # positions of labels of evidences
width = 0.25  # width of a bar
for outcome in health_outcomes:
    plt.bar(evi_pos-width/2, p_without_habit[outcome], width, label='prob without inference of habits')
    plt.bar(evi_pos+width/2, p_with_habit[outcome], width, label='prob with inference of habits')
    plt.xticks(evi_pos, labels=['bad habits', 'good habits', 'poor health', 'good health'])
    plt.xlabel(f'condition')
    plt.ylabel(f'probability of {outcome}')
    plt.legend(bbox_to_anchor=(0, 1), loc='lower left')
    plt.show()
print()


# Problem 5
print("####################################################")
print("Problem 5\n")
stroke_p5 = readFactorTablefromData(riskFactorNet, ['stroke', 'bmi', 'bp', 'cholesterol', 'smoke', 'exercise', 'diabetes'])
whole_net_p5 = [income,
            smoke, exercise, long_sit, stay_up,
            bp, cholesterol, bmi,
            diabetes_p4, stroke_p5, attack_p4, angina_p4]
hiddenVars = list(set(riskFactorNet.columns) - {'stroke', 'diabetes'})

# infer in the network in problem 4
# factor of P(stroke | diabetes=1)
stroke_diabetes_p_d1_p4 = inference(whole_net_p4, hiddenVars, ['diabetes'], [1])
# P(stroke=1 | diabetes=1)
p_d1_p4 = stroke_diabetes_p_d1_p4[stroke_diabetes_p_d1_p4['stroke']==1].iloc[0]['probs']
print("P(stroke=1 | diabetes=1) = {:.3f} in network in problem 4.".format(p_d1_p4))
# factor of P(stroke | diabetes=3)
stroke_diabetes_p_d3_p4 = inference(whole_net_p4, hiddenVars, ['diabetes'], [3])
# P(stroke=1 | diabetes=3)
p_d3_p4 = stroke_diabetes_p_d3_p4[stroke_diabetes_p_d3_p4['stroke']==1].iloc[0]['probs']
print("P(stroke=1 | diabetes=3) = {:.3f} in network in problem 4.".format(p_d3_p4))
# infer in the network in problem 5
# factor of P(stroke | diabetes=1)
stroke_diabetes_p_d1_p5 = inference(whole_net_p5, hiddenVars, ['diabetes'], [1])
p_d1_p5 = stroke_diabetes_p_d1_p5[stroke_diabetes_p_d1_p5['stroke']==1].iloc[0]['probs']
print("P(stroke=1 | diabetes=1) = {:.3f} in network in problem 5.".format(p_d1_p5))
# factor of P(stroke | diabetes=3)
stroke_diabetes_p_d3_p5 = inference(whole_net_p5, hiddenVars, ['diabetes'], [3])
p_d3_p5 = stroke_diabetes_p_d3_p5[stroke_diabetes_p_d3_p5['stroke']==1].iloc[0]['probs']
print("P(stroke=1 | diabetes=3) = {:.3f} in network in problem 5.".format(p_d3_p5))

# plot the bar plot to compare
label_pos = np.arange(2)  # positions of labels
width = 0.25  # width of a bar
plt.bar(label_pos-width/2, [p_d1_p4, p_d3_p4], width, label='prob without interaction')
plt.bar(label_pos+width/2, [p_d1_p5, p_d3_p5], width, label='prob with interaction')
plt.xticks(label_pos, labels=["P(stroke=1 | diabetes=1)", "P(stroke=1 | diabetes=3)"])
plt.ylabel(f'probability of stroke')
plt.legend()
plt.show()

print()