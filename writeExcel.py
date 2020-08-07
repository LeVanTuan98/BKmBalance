import xlsxwriter
import numpy as np
import xlwt


array = [['a1', 'a2'],
         ['a4', 'a5'],
         ['a7', 'a8'],
         ['a10', 'a11']]
# X = [4, 3, 2, 1]
# X = np.array(X).transpose()
# Y = [1 ,2 , 3, 4]
# Y = np.array(Y).transpose()
# array = [X, Y]
print(array)
col = 0

# workbook = xlwt.Workbook()
# worksheet = workbook.add_sheet("Sheet 1")
#
# # # Applying multiple styles
# # style = xlwt.easyxf('font: bold 1, color red;')
#
# worksheet.write(0, 0, ['X', 'Y'])
# for row, data in enumerate(array):
#     worksheet.write(row + 1, col, data)
#     print(row, data)
#
# workbook.save('Evaluation_Results_by_Excel/video7-8_4.xlsx')

workbook = xlsxwriter.Workbook('Evaluation_Results_by_Excel/video7-8_3.xlsx')
worksheet = workbook.add_worksheet()
col = 0
worksheet.write_row(0, 0, ['X', 'Y'])
for row, data in enumerate(array):
    worksheet.write_row(row + 1, col, data)
workbook.close()
