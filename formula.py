import pandas as pd
import numpy as np
from scipy import interpolate
import math

# Исходные данные ---------------------------------
q_of_liquid = 0.022
q_of_gas = 0.0214
dencity_of_liquid = 762.64
dencity_of_gas = 80

epsilon = 0.000018288
inner_diameter = 0.152
viscosity_of_liquid = 0.00097
g = 9.8
superficial_tencion_of_liquid = 0.00841
# -------------------------------------------------


# Площадь поперечного сечения
Ap = 3.14 / 4 * inner_diameter ** 2
# Приведенная скорость жидкости
velocity_SL = q_of_liquid / Ap
# Приведенная скорость газа
velocity_SG = q_of_gas / Ap
# Приведенная скорость смеси
velocity_M = velocity_SL + velocity_SG

# Безразмерные группы велиичин, предложенные Дансом и Россом
# Показатель скорости жидкости
Nlv = velocity_SL * (dencity_of_liquid / g / superficial_tencion_of_liquid) ** 0.25

# Показатель скорости газа
Ngv = velocity_SG * (dencity_of_liquid / g / superficial_tencion_of_liquid) ** 0.25

# Показатель диаметра трубы
Nd = inner_diameter * (dencity_of_liquid * g / superficial_tencion_of_liquid)**0.5

# Показатель вязкости жидкости
Nl = viscosity_of_liquid * (g / dencity_of_liquid / (superficial_tencion_of_liquid) ** 3) ** 0.25

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


def define_S(mode, Nl, Nd, Nlv):
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

	return f


def calc_grad_fric(mode, f):
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


Ngv_B_or_S, Ngv_S_or_Tr, Ngv_Tr_or_M = boundaries_of_modes(Nlv, Nd)
mode = define_fp(Ngv_B_or_S, Ngv_S_or_Tr, Ngv_Tr_or_M, Ngv)
S = define_S(mode, Nl, Nd, Nlv)
Vs = define_Vs(S, dencity_of_liquid, g, superficial_tencion_of_liquid)
Hliquid = Hl(Vs, velocity_SL, velocity_M, q_of_liquid)

grad_grav = calc_grad_grav(Hliquid, dencity_of_liquid, dencity_of_gas)

f = koef_of_friction(mode, velocity_SL, velocity_SG, inner_diameter, epsilon, Nd)
grad_friction = calc_grad_fric(mode, f)

grad_pressure = calc_grad(grad_friction, grad_grav)

def calc_pressure():
	""" Функция, определяющая давление в трубе путем интегрирования градиента """
	pass

print(grad_pressure)