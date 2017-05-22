from entropy import *

a = np.array([[.15,.03,.05,.07],[.02,.05,.03,.05],[.03,.2,.02,.1],[.05,.02,.1,.03]])
x = np.array([.25,.3,.2,.25])
y = np.array([.3,.15,.35,.2])

print '1.1 Entropy of X subject to P(X): ' + str(entropy(x,4))
print '1.2 Entropy of Y subject to P(Y): ' + str(entropy(y,4))
print '1.3 Conditional Entropy of X subject to P(X|Y): ' + str(conditional_entropy(a,4,4))
print '1.4 Conditional Entropy of Y subject to P(Y|X): ' + str(conditional_entropy(a.T,4,4))
print '1.5 Mutual Entropy of X and Y subject to P(Y,X): ' + str(mutual_entropy(a,4,4))

