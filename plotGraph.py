import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import xlrd

# Give the location of the file
filename_X = "Evaluation_Results_by_Excel/day21-09/verDir.xlsx"
filename_Y = "Evaluation_Results_by_Excel/day21-09/honDir.xlsx"

# To open Workbook
wb_X = xlrd.open_workbook(filename_X)
sheet_X = wb_X.sheet_by_index(0)

wb_Y = xlrd.open_workbook(filename_Y)
sheet_Y = wb_Y.sheet_by_index(0)
x_data = []
y_data = []
# For row 0 and column 0
data_size = sheet_X.nrows if (sheet_X.nrows < sheet_Y.nrows) else sheet_Y.nrows
for i in range(1, data_size):
    x_data.append(sheet_X.cell_value(i, 1))
    y_data.append(sheet_Y.cell_value(i, 1))

print(x_data)
print(y_data)

fig, ax = plt.subplots()


ax = plt.axis([0, 15, 0, 15])

center_point, = plt.plot(x_data[0], y_data[0], 'ro')

def animate(i):
    center_point.set_data(x_data[i], y_data[i])
    return center_point,

# create animation using the animate() function
myAnimation = animation.FuncAnimation(fig, func=animate, frames=range(len(x_data)), \
                                      interval=100, blit=True, repeat=True)

plt.show()

