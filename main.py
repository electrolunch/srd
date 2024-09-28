
# import numpy as np
# wdpath = r'C:\AnacondaProjects\DataManager'
# import sys
# sys.path.append(wdpath)
# # sys.path.append(r'C:\AnacondaProjects\FreqHandling')
# import numpy as np
# from UMathTools import UMath 
# import scipy.optimize
# import pandas as pd
# sys.path.append(wdpath)
# import math
# from PlotWrap import GraphTools as gt
# import allantools
# from scipy import signal
# from FileReaderModule import FileReader as Fl
import matplotlib.pyplot as plt
from polyfit import PolynomRegressor, Constraints
import pandas as pd

# from sklearn.preprocessing import PolynomialFeatures,StandardScaler, MinMaxScaler
# from sklearn.linear_model import Ridge, Lasso, ElasticNet
#%%
excel_path=r"D:\temp\srd.xlsx"

df = pd.read_excel(excel_path, names=['Vrev', 'Crev', 'Qrev', '4', '5', '6', '7'], header=0)

# scaler = StandardScaler()
# x = np.arange(1, len(y)+1, 1)
# x1=scaler.fit_transform(x.reshape(-1, 1))

polyestimator = PolynomRegressor(deg=16)
monotone_constraint = Constraints(monotonicity='inc',gridpoints=27,curvature = 'convex')
polyestimator.fit(df['Vrev'].values.reshape(-1, 1), df['Crev'].values, constraints={0: monotone_constraint}, verbose=True)

ya=polyestimator.predict(df['Vrev'].values.reshape(-1, 1))

plt.figure(figsize=(10, 6))
plt.scatter(df['Vrev'], df['Crev'], color='blue', label='Original Data')
plt.plot(df['Vrev'], ya, color='red', label='Approximation')
plt.xlabel('Vrev')
plt.ylabel('Crev')
plt.title('Original Data vs. Approximation')
plt.legend()
plt.show()

coefs = polyestimator.coeffs_
# array([ 1.07050878e+00,  4.58309582e-01,  3.73927618e-01,  1.79947258e-01,
#         5.05053240e-02,  8.39418294e-03,  7.85497372e-04,  3.09316538e-05,
#        -7.89290475e-07, -8.84097732e-08,  1.94578793e-09,  2.24871191e-10,
#        -8.33177107e-12, -5.68703086e-13,  3.07353686e-14,  2.43540202e-15,
#         4.19057130e-17])



