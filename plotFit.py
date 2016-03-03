import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn #optional
import statsmodels.api as sm

boston = pd.read_csv('boston.csv', index_col = 0)
features = [col for col in boston.columns if col != 'medv']   # every column but response column
fit3 = sm.OLS(boston.medv,sm.add_constant(boston[features]))
results3 = fit3.fit()
print(results3.summary())


def plotFit(fit):
  """Create's the 2x2 panel of plots that plot(fit) would create in R"""
  resid = fit.resid
  mu = resid.mean()
  std = resid.std(axis=0)

  #had to write my own normalize function
  def _normalize(resid):
      return (resid-mu)/std
  norm_resid = resid.apply(_normalize)


  f, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, sharex='col', sharey='row')

  ax1.scatter(fit.fittedvalues, fit.resid)
  ax1.set_xlabel('Fitted Values')
  ax1.set_ylabel('Residuals')
  ax1.set_title('Residuals vs Fitted')


  sm.qqplot(fit.resid, ax=ax2)
  ax2.set_title('QQ plot')

  ax3.scatter(fit.fittedvalues, norm_resid)
  ax3.set_xlabel('Fitted Values')
  ax3.set_ylabel('Standardized Residuals')
  ax3.set_title('Scale-Location')

  sm.graphics.influence_plot(fit, ax=ax4, criterion="cooks")

  plt.show()


# ACTUAL CALL
plotFit(results3)