# Plotting state-of-health values with a trend
#  
# Positive current (A) indicator descriptions
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

DIR = "../Logsn/ind_and_selBcol/v140/"
FILE = DIR + "JPmth023.csv" #a
FILE2 = DIR + "JPmth044.csv" #b
FILE3 = DIR + "JPmth056.csv" #c 
FILE4 = DIR + "JPmth067.csv" #d 
FILE5 = DIR + "JPmth071.csv" #e

# Read df 
dfA = pd.read_csv(FILE)
dfB = pd.read_csv(FILE2)
dfC = pd.read_csv(FILE3)
dfD = pd.read_csv(FILE4)
dfE = pd.read_csv(FILE5)

if (len(dfA) != len(dfB)) or (len(dfA) != len(dfC)
    or len(dfA) != len(dfD)) or (len(dfA) != len(dfE)):
    print("Mismatch in data length!")

# Define x-axis for sns plot
dfA["month"] = range(dfA.shape[0])

# Remove headers
#dfA.columns = range(dfA.shape[1])

# create scatter plot after ensuring the same x-axis; order = 1 is linear trend
sns.regplot(x=dfA.month, y=dfA.SOH, marker="+", order= 1, ci=None, label="battery a")
sns.regplot(x=dfA.month, y=dfB.SOH, marker="2", order= 1, ci=None, label="battery b")
sns.regplot(x=dfA.month, y=dfC.SOH, marker="*", order= 1, ci=None, label="battery c")
sns.regplot(x=dfA.month, y=dfD.SOH, marker="x", order= 1, ci=None, label="battery d")
sns.regplot(x=dfA.month, y=dfE.SOH, marker="4", order= 1, ci=None, label="battery e")

# legend and title
plt.legend()
plt.title('Linear trend for the first 32 months of SoH; 5 battery packs', y= 1, loc = 'right')

# save figure
plt.savefig('Fig_SoH_trend5.pdf')

# show plot
plt.show()


