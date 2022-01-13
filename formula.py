import pandas as pd
import numpy as np
from scipy import interpolate
import math

from gas_properties import *
from add_functions import *

# Исходные данные ---------------------------------
q_of_liquid = 0.022 # м3/с
q_of_gas_surface = 200 # тыс.м3/сут
dencity_of_liquid = 762.64 #кг/м3
dencity_of_gas = 80 #кг/м3

epsilon = 0.000018288
inner_diameter = 0.152
viscosity_of_liquid = 0.00097
g = 9.8
superficial_tencion_of_liquid = 0.00841
# -------------------------------------------------

# Площадь поперечного сечения
Ap = 3.14 / 4 * inner_diameter ** 2




def bezrazmernie_pokazateli(velocity_SL,velocity_SG,inner_diameter, viscosity_of_liquid , dencity_of_liquid, g, superficial_tencion_of_liquid):
	# Безразмерные группы велиичин, предложенные Дансом и Россом
	# Показатель скорости жидкости
	Nlv = velocity_SL * (dencity_of_liquid / g / superficial_tencion_of_liquid) ** 0.25

	# Показатель скорости газа
	Ngv = velocity_SG * (dencity_of_liquid / g / superficial_tencion_of_liquid) ** 0.25

	# Показатель диаметра трубы
	Nd = inner_diameter * (dencity_of_liquid * g / superficial_tencion_of_liquid)**0.5

	# Показатель вязкости жидкости
	Nl = viscosity_of_liquid * (g / dencity_of_liquid / (superficial_tencion_of_liquid) ** 3) ** 0.25

	return Nlv, Ngv, Nd, Nl

# ---------------------------------------------

# Nlv = 11.87
# Ngv = 11.54
# Nd = 143.8
# Nl = 0.0118

# Определение границ режимов
def boundaries_of_modes(Nlv, Nd):
	"""
	Функция, определяющая числовые значения границ режимов
	"""
	data = pd.read_excel(r"data.xlsx", sheet_name='L1-L2')

	x_array = np.array(data.L1_x)
	x = x_array[~np.isnan(x_array)]
	y_array = np.array(data.L1_y)
	y = y_array[~np.isnan(y_array)]
	f_L1 = interpolate.interp1d(x, y, kind=2)

	x_array = np.array(data.L2_x)
	x = x_array[~np.isnan(x_array)]
	y_array = np.array(data.L2_y)
	y = y_array[~np.isnan(y_array)]
	f_L2 = interpolate.interp1d(x, y, kind=2)

	L1 = f_L1(Nd)
	L2 = f_L2(Nd)

	# Граница пузырькового/пробкового режима
	Ngv_B_or_S = L1 + L2 * Nlv

	# Граница пробкового/переходного режима
	Ngv_S_or_Tr = 50 + 36 * Nlv

	# Граница переходного/эмульсионного режима
	Ngv_Tr_or_M = 75 + 84 * Nlv ** 0.75

	return Ngv_B_or_S, Ngv_S_or_Tr, Ngv_Tr_or_M


# Определение режима
def define_fp(Ngv_B_or_S, Ngv_S_or_Tr, Ngv_Tr_or_M, Ngv):
	""" Функция, определяющая структуру потока """
	# Пузырьковый режим
	if Ngv <= Ngv_B_or_S:
		return 1
	# Пробковый режим
	elif Ngv <= Ngv_S_or_Tr:
		return 2
	# Переходный режим
	elif Ngv <= Ngv_Tr_or_M:
		return 3
	# Эмульсионный режим
	else:
		return 4


def define_S(mode, Nl, Nd, Nlv, Ngv):
	"""
	Определение скорости проскальзывания
	:param mode: Тип режима, определяемый в функции define_fp
	:return S: Значение скорости проскальзывания Vs
	"""
	data = pd.read_excel(r"data.xlsx", sheet_name='F1-F7')

	# Пузырьковый режим
	if mode == 1:
		x_array = np.array(data.F1_x)
		x = x_array[~np.isnan(x_array)]
		y_array = np.array(data.F1_y)
		y = y_array[~np.isnan(y_array)]
		f_F1 = interpolate.interp1d(x, y, kind=2)

		x_array = np.array(data.F2_x)
		x = x_array[~np.isnan(x_array)]
		y_array = np.array(data.F2_y)
		y = y_array[~np.isnan(y_array)]
		f_F2 = interpolate.interp1d(x, y, kind=2)

		x_array = np.array(data.F3_x)
		x = x_array[~np.isnan(x_array)]
		y_array = np.array(data.F3_y)
		y = y_array[~np.isnan(y_array)]
		f_F3 = interpolate.interp1d(x, y, kind=2)

		x_array = np.array(data.F4_x)
		x = x_array[~np.isnan(x_array)]
		y_array = np.array(data.F4_y)
		y = y_array[~np.isnan(y_array)]
		f_F4 = interpolate.interp1d(x, y, kind=2)

		F1 = f_F1(Nl)
		F2 = f_F2(Nl)
		F3 = f_F3(Nl)
		F4 = f_F4(Nl)

		F3_corr = F3 - F4 / Nd

		S = F1 + F2 * Nlv + F3_corr * (Ngv / (1 + Nlv)) ** 2

	# Пробковый режим
	elif mode == 2:
		x_array = np.array(data.F5_x)
		x = x_array[~np.isnan(x_array)]
		y_array = np.array(data.F5_y)
		y = y_array[~np.isnan(y_array)]
		f_F5 = interpolate.interp1d(x, y, kind=2)

		x_array = np.array(data.F6_x)
		x = x_array[~np.isnan(x_array)]
		y_array = np.array(data.F6_y)
		y = y_array[~np.isnan(y_array)]
		f_F6 = interpolate.interp1d(x, y, kind=2)

		x_array = np.array(data.F7_x)
		x = x_array[~np.isnan(x_array)]
		y_array = np.array(data.F7_y)
		y = y_array[~np.isnan(y_array)]
		f_F7 = interpolate.interp1d(x, y, kind=2)

		F5 = f_F5(Nl)
		F6 = f_F6(Nl)
		F7 = f_F7(Nl)

		F6_corr = 0.029 * Nd + F6

		S = (1 + F5) * (Ngv ** 0.982 + F6_corr) / (1 + F7 * Nlv)

	# Переходный режим
	elif mode == 3:
		# A = (Ngv_Tr_or_M - Ngv) / (Ngv_Tr_or_M - Ngv_S_or_Tr)
		S = -1

	# Эмульсионный режим
	elif mode == 4:
		S = 0

	return S


def define_Vs(S, dencity_of_liquid, g, superficial_tencion_of_liquid):
	""" Функция, возвращающая значение скорости проскальзывания """
	if S == 0:
		Vs = 0
	elif S == -1:
		Vs = 0  # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
	else:
		Vs = S / (dencity_of_liquid / g / superficial_tencion_of_liquid) ** 0.25

	return Vs


def Hl(velocity_S, velocity_SL, velocity_M, q_of_liquid):
	""" Функция, определяющая объемное содержание жидкости """
	if velocity_S == 0:
		Hl = q_of_liquid
	else:
		Hl = (velocity_S - velocity_M + ((velocity_M - velocity_S) ** 2 + 4 * velocity_S * velocity_SL) ** 0.5) / (2 * velocity_S)

	return Hl


def calc_grad_grav(Hl, dencity_of_liquid, dencity_of_gas):
	""" Функция, определяющая градиент давления, обусловленного гравитацией """
	dencity_of_mixture = dencity_of_liquid * Hl + dencity_of_gas * (1 - Hl)
	return dencity_of_mixture * 9.81


def koef_of_friction(mode, velocity_SL, velocity_SG, inner_diameter, epsilon, Nd):
	""" Функция для определения коэффициента трения"""
	data = pd.read_excel(r"data.xlsx", sheet_name='f2_friction')

	if mode == 1 or mode == 2:
		x_array = np.array(data.Vertical_x)
		x = x_array[~np.isnan(x_array)]
		y_array = np.array(data.Vertical_y)
		y = y_array[~np.isnan(y_array)]
		f2_vertical = interpolate.interp1d(x, y, kind=2)

		x_array = np.array(data.Horizontal_x)
		x = x_array[~np.isnan(x_array)]
		y_array = np.array(data.Horizontal_y)
		y = y_array[~np.isnan(y_array)]
		f2_horizontal = interpolate.interp1d(x, y, kind=2)

		# Для диаграммы Муди
		# N_Re_L = dencity_of_liquid * velocity_SL * inner_diameter / viscosity_of_liquid
		# N_Re_G = dencity_of_gas * velocity_SG * inner_diameter / viscosity_of_gas

		f1 = (1 / (1.74 - 2 * math.log10(2 * epsilon / inner_diameter))) ** 2

		koef_for_f2 = f1 * velocity_SG * Nd ** (2 / 3) / (4 * velocity_SL)

		f2 = f2_vertical(koef_for_f2)

		f3 = 1 + f1 / 4 * (velocity_SG / 50 / velocity_SL) ** 0.5

		f = f1 * f2 / f3

	elif mode == 4:
		# Для диаграммы Муди
		# N_Re_G = dencity_of_gas * velocity_SG * inner_diameter / viscosity_of_gas
		f = 0.01
	else:
		f = 0.001

	return f


def calc_grad_fric(mode, f, velocity_SL, velocity_SG, velocity_M):
	""" Функция, вычисляющая градиент давления на трение """
	if mode == 1 or mode == 2:
		grad_fric = f * dencity_of_liquid * velocity_SL * velocity_M / (2 * inner_diameter)
	elif mode == 4:
		grad_fric = f * dencity_of_gas * velocity_SG ** 2 / (2 * inner_diameter)
	else:
		grad_fric = 1

	return grad_fric


def calc_grad(grad_friction, grad_grav):
	""" Функция, вычисляющая общий градиент давления, в атм/м """
	grad_pressure = (grad_friction + grad_grav) / 101325

	return grad_pressure


def calc_pressure():
	""" Функция, определяющая давление в трубе путем интегрирования градиента """
	pass

def toDefinePressureOnDepth(wellheadPressure, gradientOfPressure, depth):
	"""
    Функция для расчета давления на определенной глубине
    :param wellheadPressure: Устьевое давление, [бар]
    :param gradientOfPressure: Градиент давления, [Па/м]
    :param depth: Текущая глубина, [м]
    :return: Значение давления на текущей глубине, [бар]
    """
	return wellheadPressure + (gradientOfPressure / 100000) * depth


def calc_pressure(q_of_liquid, top_hole_pressure = 'None'):
	relative_density_gas = 0.4

	pressure_list, depth_list = [], []

	# 100 интервалов по длине скважиине
	for i in range(0,101):

		if i == 0:
			if top_hole_pressure == 'None':
				# Задаю на выбор вручную, бар
				bottom_hole_pressure = 50
				current_pressure = bottom_hole_pressure
			else:
				current_pressure = top_hole_pressure
		current_temperature = 373

		# Длина интервала = 20 м
		current_depth = i * 20

		currentKoefZ = toDefineKoefZ(current_pressure, relative_density_gas, current_temperature)
		Bg = 3.511 * 0.001 * currentKoefZ * current_temperature / current_pressure

		q_of_gas = (q_of_gas_surface * 1000) / 86400 * Bg

		# Приведенная скорость жидкости
		velocity_SL = q_of_liquid / Ap

		# Приведенная скорость газа
		velocity_SG = q_of_gas / Ap

		# Приведенная скорость смеси
		velocity_M = velocity_SL + velocity_SG

		Nlv, Ngv, Nd, Nl = bezrazmernie_pokazateli(velocity_SL,velocity_SG,inner_diameter, viscosity_of_liquid , dencity_of_liquid, g, superficial_tencion_of_liquid)


		Ngv_B_or_S, Ngv_S_or_Tr, Ngv_Tr_or_M = boundaries_of_modes(Nlv, Nd)
		mode = define_fp(Ngv_B_or_S, Ngv_S_or_Tr, Ngv_Tr_or_M, Ngv)
		S = define_S(mode, Nl, Nd, Nlv, Ngv)
		Vs = define_Vs(S, dencity_of_liquid, g, superficial_tencion_of_liquid)
		Hliquid = Hl(Vs, velocity_SL, velocity_M, q_of_liquid)

		grad_grav = calc_grad_grav(Hliquid, dencity_of_liquid, dencity_of_gas)

		f = koef_of_friction(mode, velocity_SL, velocity_SG, inner_diameter, epsilon, Nd)
		grad_friction = calc_grad_fric(mode, f, velocity_SL, velocity_SG, velocity_M)

		grad_pressure = calc_grad(grad_friction, grad_grav)

		if top_hole_pressure == 'None':
			current_pressure = bottom_hole_pressure - (grad_pressure / 100000) * current_depth
		else:
			current_pressure = top_hole_pressure + (grad_pressure / 100000) * current_depth

		depth_list.append(current_depth)
		pressure_list.append(current_pressure)

	# pressureList = np.array(pressureList)
	# depthList_array = np.array(depthList)
	# drawing(pressureList, depthList_array)

	return current_pressure, depth_list, pressure_list


# pressure_list = []
# for i in [100,125,150,175,200]:
# 	q_of_liquid = i / 86400
# 	current_pressure = calc_pressure(q_of_liquid, top_hole_pressure=20)
# 	pressure_list.append(current_pressure)
#
# drawing([100,125,150,175,200], pressure_list, type_graph= 'VLP')

# Дебит жидкости 150 м3/сут
q_of_liquid = 150 / 86400
current_pressure, depth_this_iter, pressure_this_iter = calc_pressure(q_of_liquid, top_hole_pressure = 60)

pressureList = np.array(pressure_this_iter)
depthList_array = np.array(depth_this_iter)
drawing(pressureList, depthList_array)




