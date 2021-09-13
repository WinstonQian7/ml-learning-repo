#Identifying Gaussian distribution for univariate data!
#Choose a variable to analyze in place of x 
from matplotlib import pyplot
from statsmodels.graphics.gofplots import qqplot
from scipy.stats import shapiro
from scipy.stats import normaltest
from scipy.stats import anderson
from numpy.random import seed
seed(1)
#Histogram
pyplot.hist(x)
#QQPlot
qqplot(x,line='s')
#Shapiro-Wilk Test
stat,p = shapiro(x)
alpha = 0.05
if p > alpha:
  print('Fail to reject null hypothesis H0')
else:
  print('Reject null hypothesis H0')
#D'Agostino's K^2 Test - kurtosis and skewness
stat, p = normaltest(x_train)
if p > alpha:
  print('Fail to reject null hypothesis H0') #normal dis
else:
  print('Reject null hypothesis H0') #not normal dis
#Anderson-Darling Test
result = anderson(x)
p = 0
for i in range(len(result.critical_values)):
	sl, cv = result.significance_level[i], result.critical_values[i]
	if result.statistic < result.critical_values[i]:
		print('Fail to reject null hypothesis H0')
	else:
		print('Reject null hypothesis H0')