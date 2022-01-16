import matplotlib.pyplot as plt
from scipy import interpolate
import pandas as pd
import numpy as np

def drawing(*args, type_graph='profileOfPressure'):
	x, y = args
	if type_graph =='profileOfPressure':
		# Построение графика
		plt.title("Профиль давления")  # заголовок
		plt.xlabel("P, бар")  # ось абсцисс
		plt.ylabel("Глубина, м")  # ось ординат
		plt.grid()  # включение отображение сетки
		plt.plot(x, y)  # построение графика
		plt.show()

	if type_graph == 'VLP':
		plt.title("VLP")  # заголовок
		plt.xlabel("Q, м3/сут")  # ось абсцисс
		plt.ylabel("P, бар")  # ось ординат
		plt.grid()  # включение отображение сетки
		plt.plot(x, y)  # построение графика
		plt.show()

	plt.show()


def data_of_moody(epsilon, diam, numberOfReinolds):

	if numberOfReinolds <= 2000:
		koef_friction = 64 / numberOfReinolds
	else:
		data = pd.read_excel(r"data.xlsx", sheet_name='moody')
		epsilon_d = epsilon / diam

		if epsilon_d <= 1*10**(-6):
			tube = 'tube1'
		elif epsilon_d <= 2*10**(-6):
			tube = 'tube2'
		elif epsilon_d <= 5*10**(-6):
			tube = 'tube3'
		elif epsilon_d <= 1 * 10 ** (-5):
			tube = 'tube4'
		elif epsilon_d <= 2*10**(-5):
			tube = 'tube5'
		elif epsilon_d <= 5*10**(-5):
			tube = 'tube6'
		elif epsilon_d <= 0.0001:
			tube = 'tube7'
		elif epsilon_d <= 0.0002:
			tube = 'tube8'
		elif epsilon_d <= 0.0005:
			tube = 'tube9'
		elif epsilon_d <= 0.001:
			tube = 'tube10'
		elif epsilon_d <= 0.002:
			tube = 'tube11'
		elif epsilon_d <= 0.005:
			tube = 'tube12'
		elif epsilon_d <= 0.01:
			tube = 'tube13'
		elif epsilon_d <= 0.015:
			tube = 'tube14'
		elif epsilon_d <= 0.02:
			tube = 'tube15'
		elif epsilon_d <= 0.03:
			tube = 'tube16'
		elif epsilon_d <= 0.04:
			tube = 'tube17'
		elif epsilon_d <= 0.05:
			tube = 'tube18'

		axes_x = tube + '_x'
		axes_y = tube + '_y'

		x_array = np.array(data[axes_x])
		x = x_array[~np.isnan(x_array)]
		y_array = np.array(data[axes_y])
		y = y_array[~np.isnan(y_array)]

		f_L1 = interpolate.interp1d(x, y, kind=2)

		koef_friction = f_L1(numberOfReinolds)

	return koef_friction
