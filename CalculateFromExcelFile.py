import xlrd
import numpy as np
import matplotlib.pyplot as plt

# Give the location of the file
filename = "Evaluation Results/video13-6.xlsx"

# To open Workbook
wb = xlrd.open_workbook(filename)
sheet = wb.sheet_by_index(0)
x = []
y = []
# For row 0 and column 0
for i in range(1, sheet.nrows):
    x.append(sheet.cell_value(i, 0))
    y.append(sheet.cell_value(i, 1))

x_avr = round(np.average(x), 2)
y_avr = round(np.average(y), 2)

X = x - x_avr
Y = y - y_avr
n = len(X)
print("maximum X: ", round(max(X), 2))
print("minimum X: ", round(min(X), 2))


# Resultant distance (RD)
RD = np.sqrt(X**2, Y**2)

# Mean Distance (MD)
MD = np.average(RD)

# Root mean square distance (RMS dis)
RMS = np.sqrt((np.sum(RD**2))/n)

# Total Path
total_path = 0
for i in range(n - 1):
    total_path += np.sqrt((X[i + 1] - X[i])**2 + (Y[i + 1] - Y[i])**2)

# Mean Velocity
times = n/30
MV = round(total_path/times, 2)
print("MV: ", MV)

time = np.arange(0, n, 1)/30

plt.plot(time, X, color='green', linestyle='dashed', linewidth=1,
         marker='o', markerfacecolor='red', markersize=5)

plt.ylim((-2, 2))
plt.xlabel('Time(s)')
plt.ylabel('Distance(cm)')
plt.grid(True)

plt.title('The Chart shows the change of center point')
plt.show()

