from sklearn.model_selection import GridSearchCV
from sklearn.metrics import make_scorer
from sklearn.metrics import f1_score

parameters = {'kernel':['poly', 'rbf'],'C':[0.1, 1, 10]}
scorer = make_scorer(f1_score)

grid_obj = GridSearchCV(clf, parameters, scoring=scorer)
grid_fit = grid_obj.fit(X, y)

best_clf = grid_fit.best_estimator_ #clf = classifier
