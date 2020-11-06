# Positive current (A) indicator descriptions
import numpy 
import pandas as pd
from pandas import set_option
from matplotlib import pyplot
from pandas.plotting import register_matplotlib_converters
register_matplotlib_converters()

DIR = "../Logsn/ind_and_selBcol/v140/"
FILE = DIR + "JPmth023.csv" #a
FILE2 = DIR + "JPmth044.csv" #b
FILE3 = DIR + "JPmth056.csv" #c 
FILE4 = DIR + "JPmth067.csv" #d 
FILE5 = DIR + "JPmth071.csv" #e

datasetA = pd.read_csv(FILE, usecols= ['BatteryStateOfCharge_Percent','BatteryVoltage_V',
'A_mean', 'min', 'Wh_sum', 'DSOC','DV', 'fD_all', 'fD_sel',
'cyc', 'TemperatureEnvironment_C', 'SOH'])
datasetB = pd.read_csv(FILE2, usecols= ['BatteryStateOfCharge_Percent','BatteryVoltage_V',
'A_mean', 'min', 'Wh_sum', 'DSOC','DV', 'fD_all', 'fD_sel',
'cyc', 'TemperatureEnvironment_C', 'SOH'])
datasetC = pd.read_csv(FILE3, usecols= ['BatteryStateOfCharge_Percent','BatteryVoltage_V',
'A_mean', 'min', 'Wh_sum', 'DSOC','DV', 'fD_all', 'fD_sel',
'cyc', 'TemperatureEnvironment_C', 'SOH'])
datasetD = pd.read_csv(FILE4, usecols= ['BatteryStateOfCharge_Percent','BatteryVoltage_V',
'A_mean', 'min', 'Wh_sum', 'DSOC','DV', 'fD_all', 'fD_sel',
'cyc', 'TemperatureEnvironment_C', 'SOH'])
datasetE = pd.read_csv(FILE5, usecols= ['BatteryStateOfCharge_Percent','BatteryVoltage_V',
'A_mean', 'min', 'Wh_sum', 'DSOC','DV', 'fD_all', 'fD_sel',
'cyc', 'TemperatureEnvironment_C', 'SOH'])

if (len(datasetA) != len(datasetB)) or (len(datasetA) != len(datasetC)
    or len(datasetA) != len(datasetD)) or (len(datasetA) != len(datasetE)):
    print("Mismatch in data length!")

# Set x-axis scale label
x_axis_scale = range(0, len(datasetA)+1, 5)

# all charging: state of health
name = ['SOH'] 
l1, = pyplot.plot(datasetA[name], 'C0-', label = "battery a")
l2, = pyplot.plot(datasetB[name], 'C1--', label = "battery b")
l3, = pyplot.plot(datasetC[name], 'C2-.', label = "battery c")
l4, = pyplot.plot(datasetD[name], 'C3:', label = "battery d")
l5, = pyplot.plot(datasetE[name], 'C4-', label = "battery e")
pyplot.legend(handles=[l1,l2,l3,l4,l5])
pyplot.title('State of health (SoH)', y= 1, loc = 'right')
pyplot.xticks(x_axis_scale)
pyplot.xlabel('Time (months)')
pyplot.ylabel('SoH (%)')
# ambient temperature
#pyplot.figtext(0.2, 0.2, 'Ambient temperature %.1f-%.1f C' %(minC, maxC), style='italic',
#        bbox={'facecolor': 'white', 'alpha': 0.5, 'pad': 10})
pyplot.savefig('Fig_SoH_5.pdf')
pyplot.show()

# all charging: charging cycles
name = ['cyc']
l1, = pyplot.plot(datasetA[name], 'C0-', label = "battery a")
l2, = pyplot.plot(datasetB[name], 'C1--', label = "battery b")
l3, = pyplot.plot(datasetC[name], 'C2-.', label = "battery c")
l4, = pyplot.plot(datasetD[name], 'C3:', label = "battery d")
l5, = pyplot.plot(datasetE[name], 'C4-', label = "battery e")
pyplot.legend(handles=[l1,l2,l3,l4,l5])
pyplot.title('Charging cycles', y=1, loc='right')
pyplot.xticks(x_axis_scale)
pyplot.xlabel('Time (months)')
pyplot.ylabel('Full charging cycles')
pyplot.savefig('Fig_cycles_5.pdf')
pyplot.show()

# environment temperature 
name = ['TemperatureEnvironment_C']
pyplot.plot(datasetA[name])
pyplot.title('Battery a ambient temperature', y=1, loc='right')
pyplot.xticks(x_axis_scale)
pyplot.xlabel('Time (months)')
pyplot.ylabel('Ambient temperature ($^\circ$C)')
pyplot.savefig('Fig_temp_A.pdf')
pyplot.show()
