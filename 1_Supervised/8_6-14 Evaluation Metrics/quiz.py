#Accuracy 8_6
total = 14
correct = 11

accuracy = 11/14
print(accuracy*100)

#Precision 8_11
diagnosed_positive = 8
positive = 6
negative = 2

precision = 6/8
print(precision)

#Recall 8_12
actual_positive = 7
predicted_positive = 6

recall= predicted_positive/actual_positive
print(recall)

#F1 8_13
precision = 0.556
recall = 0.833
#Issue of average(x+y)/2: If one metric is really bad and the other one good, the average will be ok.
# We want to punish the f1 if even one metric is bad, no matter the other one

# Harmonic mean (2xy/x+y) solves this:
f1 = 2 * (precision*recall)/(precision+recall)
print(f1*100)

#Jupyter Notebook
a = "recall"
b = "precision"
c = "accuracy"
d = 'f1-score'


seven_sol = {
'We have imbalanced classes, which metric do we definitely not want to use?': c,
'We really want to make sure the positive cases are all caught even if that means we identify some negatives as positives': a,    
'When we identify something as positive, we want to be sure it is truly positive': b, 
'We care equally about identifying positive and negative cases': d    
}

#That's right!  Naive Bayes was the best model for all of our metrics except precision!

#+++ACCURACY+++
#Naive Bayes: 0.988513998564
#Bagging: 0.974874371859
#Random Forest: 0.982770997846
#Ada Boost: 0.977027997128
#SVM: 0.867193108399
#+++PRECISION+++
#Naive Bayes: 0.972067039106
#Bagging: 0.912087912088
#Random Forest: 1.0
#Ada Boost: 0.969325153374
#SVM: 0.0
#+++RECALL+++
#Naive Bayes: 0.940540540541
#Bagging: 0.897297297297
#Random Forest: 0.87027027027
#Ada Boost: 0.854054054054
#SVM: 0.0
#+++F1+++
#Naive Bayes: 0.956043956044
#Bagging: 0.904632152589
#Random Forest: 0.93063583815
#Ada Boost: 0.908045977011
#SVM: 0.0