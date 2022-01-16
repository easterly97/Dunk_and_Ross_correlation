import pandas as pd
import numpy as np
from scipy import interpolate
import math

from liquid_properties import *
from add_functions import *


def bezrazmernie_pokazateli(velocity_SL, velocity_SG, inner_diameter, viscosity_of_liquid, dencity_of_liquid, g,
							superficial_tencion_of_liquid):
	# Безразмерные группы велиичин, предложенные Дансом и Россом
	# Показатель скорости жидкости
	Nlv = velocity_SL * (dencity_of_liquid / g / superficial_tencion_of_liquid) ** 0.25

	# Показатель скорости газа
	Ngv = velocity_SG * (dencity_of_liquid / g / superficial_tencion_of_liquid) ** 0.25

	# Показатель диаметра трубы
	Nd = inner_diameter * (dencity_of_liquid * g / superficial_tencion_of_liquid) ** 0.5

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
	else:
		Vs = S / (dencity_of_liquid / g / superficial_tencion_of_liquid) ** 0.25

	return Vs


def define_Hl(velocity_S, velocity_SL, velocity_M, q_of_liquid, q_of_gas):
	""" Функция, определяющая объемное содержание жидкости """
	# Если режим - эмульсионный
	if velocity_S == 0:
		lambda_l = q_of_liquid / (q_of_liquid + q_of_gas)
		Hl = lambda_l
	# Если режим пузырьковый или снарядный
	else:
		Hl = (velocity_S - velocity_M + ((velocity_M - velocity_S) ** 2 + 4 * velocity_S * velocity_SL) ** 0.5) / (
				2 * velocity_S)

	return Hl


def calc_grad_grav(dencity_of_mixture):
	""" Функция, определяющая градиент давления, обусловленного гравитацией """
	return dencity_of_mixture * 9.81


def koef_of_friction(mode, velocity_SL, velocity_SG, inner_diameter, epsilon, Nd, dencity_of_liquid,
					 viscosity_of_liquid):
	""" Функция для определения коэффициента трения"""
	data = pd.read_excel(r"data.xlsx", sheet_name='f2_friction')

	# Если пузырьковый или снарядный режим
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
		# N_Re_G = dencity_of_gas * velocity_SG * inner_diameter / viscosity_of_gas
		# f1 = (1 / (1.74 - 2 * math.log10(2 * epsilon / inner_diameter))) ** 2
		# С использованием диаграммы Муди
		N_Re_L = dencity_of_liquid * velocity_SL * inner_diameter / viscosity_of_liquid
		f1 = data_of_moody(epsilon, inner_diameter, N_Re_L)

		koef_for_f2 = f1 * velocity_SG * Nd ** (2 / 3) / (4 * velocity_SL)
		f2 = f2_vertical(koef_for_f2)

		f3 = 1 + f1 / 4 * (velocity_SG / 50 / velocity_SL) ** 0.5

		f = f1 * f2 / f3

	# Для эмульсионного режима
	elif mode == 4:
		N_Re_G = dencity_of_gas * velocity_SG * inner_diameter / viscosity_of_gas
		f = data_of_moody(epsilon, inner_diameter, N_Re_G)

	else:
		f = 0.001

	return f


def calc_grad_fric(mode, f, velocity_SL, velocity_SG, velocity_M):
	""" Функция, вычисляющая градиент давления на трение """
	# Для пузырькового и снарядного режима
	if mode == 1 or mode == 2:
		grad_fric = f * dencity_of_liquid * velocity_SL * velocity_M / (2 * inner_diameter)

	# Для эмульсионного режима
	elif mode == 4:
		grad_fric = f * dencity_of_gas * velocity_SG ** 2 / (2 * inner_diameter)
	# Для переходного режима
	# fixme: добавить функционал для переходного режима
	else:
		grad_fric = 1

	return grad_fric


def calc_grad_acceleration(mode, velocity_M, velocity_SG, dencity_N, pressure):
	"""
	Функция, вычисляющая градиент ускорения, в бар/м
	:param mode: Тип режима
	:param velocity_M: Скорость смеси, [м/с]
	:param velocity_SG: Скорость газа, [м/с]
	:param dencity_N: плотность смеси, [кг/м3]
	:param pressure: Текущее давление, [Па]
	:return: Значение градиента давления на ускорение, [Па]
	"""
	grad_acceleration = 0
	if mode == 1 or mode == 2:
		grad_acceleration = 0
	elif mode == 4:
		kinetic_enegry = velocity_M * velocity_SG * dencity_N / (pressure * 100000)
		grad_acceleration = kinetic_enegry

	return grad_acceleration


def calc_grad(mode, grad_friction, grad_grav, grad_acceleration):
	"""
	Функция, вычисляющая общий градиент давления, в бар/м
	:param mode: Тип режима
	:param grad_friction: Устьевое давление, [Па]
	:param grad_grav: Градиент давления, [Па]
	:param grad_acceleration: Градиент давления, [Па]
	:return: Значение общего градиента давления, [бар]
	"""
	# Для эмульсионного режима
	if mode == 4:
		grad_pressure = (grad_grav + grad_friction) / (1 - grad_acceleration)
	# Для пузырькового и пробкового режимов
	else:
		grad_pressure = (grad_friction + grad_grav + grad_acceleration)

	return grad_pressure / 100000


def toDefinePressureOnDepth(wellheadPressure, gradientOfPressure, depth):
	"""
    Функция для расчета давления на определенной глубине
    :param wellheadPressure: Устьевое давление, [бар]
    :param gradientOfPressure: Градиент давления, [Па/м]
    :param depth: Текущая глубина, [м]
    :return: Значение давления на текущей глубине, [бар]
    """
	return wellheadPressure + (gradientOfPressure / 100000) * depth


def calc_pressure(q_of_liquid_surface, bottom_hole_pressure, length_tube, step_of_calc, top_hole_pressure='None'):
	"""
	Функция, определяющая давление в трубе путем интегрирования градиента
	"""

	pressure_list, depth_list = [], []

	# Цикл по интервалам
	count_of_intervals = math.ceil(length_tube / step_of_calc)
	for i in range(0, count_of_intervals + 1):

		if i == 0:
			if top_hole_pressure == 'None':
				current_pressure = bottom_hole_pressure
			else:
				current_pressure = top_hole_pressure


		current_depth = i * step_of_calc

		# Определение температуры на определенной глубине
		if top_hole_pressure != 'None':
			current_temperature = strata_temperature - (
						strata_temperature - well_head_temperature) / length_tube * current_depth
		else:
			current_temperature = well_head_temperature + (
						strata_temperature - well_head_temperature) / length_tube * current_depth

		# Определение свойств нефти
		# Растворимость газа по корреляции Стэндинга
		Rs = 0.178 * relative_density_of_gas * ((current_pressure / 1.254 + 1.4) * 10 ** (
					0.0125 * specific_gravity_oil - 0.001638 * current_temperature - 0.02912)) ** 1.2048
		# Объемный коэффициент по корреляции Стэндинга (при давлениях ниже давления насыщения)
		Boil = 0.9759 + 0.00012 * (5.618 * Rs * (
					relative_density_of_gas / specific_gravity_oil) ** 0.5 + 2.25 * current_temperature + 40) ** 1.2

		# Определение свойств газа
		currentKoefZ = toDefineKoefZ(current_pressure, relative_density_of_gas, current_temperature)
		Bg = 3.511 * 0.001 * currentKoefZ * current_temperature / current_pressure

		q_of_liquid = q_of_liquid_surface / 86400 * Boil
		q_of_gas = (q_of_gas_surface * 1000) / 86400 * Bg

		# Приведенная скорость жидкости
		velocity_SL = q_of_liquid / Ap

		# Приведенная скорость газа
		velocity_SG = q_of_gas / Ap

		# Приведенная скорость смеси
		velocity_M = velocity_SL + velocity_SG

		Nlv, Ngv, Nd, Nl = bezrazmernie_pokazateli(velocity_SL, velocity_SG, inner_diameter, viscosity_of_liquid,
												   dencity_of_liquid, g, superficial_tencion_of_liquid)

		Ngv_B_or_S, Ngv_S_or_Tr, Ngv_Tr_or_M = boundaries_of_modes(Nlv, Nd)
		mode = define_fp(Ngv_B_or_S, Ngv_S_or_Tr, Ngv_Tr_or_M, Ngv)

		# Переходный режим = особенный случай
		if mode != 3:
			S = define_S(mode, Nl, Nd, Nlv, Ngv)
			Vs = define_Vs(S, dencity_of_liquid, g, superficial_tencion_of_liquid)
			Hliquid = define_Hl(Vs, velocity_SL, velocity_M, q_of_liquid, q_of_gas)
			dencity_of_mixture = dencity_of_liquid * Hliquid + dencity_of_gas * (1 - Hliquid)

			grad_grav = calc_grad_grav(dencity_of_mixture)

			f = koef_of_friction(mode, velocity_SL, velocity_SG, inner_diameter, epsilon, Nd, dencity_of_liquid,
								 viscosity_of_liquid)
			grad_friction = calc_grad_fric(mode, f, velocity_SL, velocity_SG, velocity_M)

			grad_acceleration = calc_grad_acceleration(mode, velocity_M, velocity_SG, dencity_of_mixture,
													   current_pressure)

			grad_pressure = calc_grad(mode, grad_friction, grad_grav, grad_acceleration)

		else:
			# Пробковый режим
			mode = 2
			S = define_S(mode, Nl, Nd, Nlv, Ngv)
			Vs = define_Vs(S, dencity_of_liquid, g, superficial_tencion_of_liquid)
			Hliquid = define_Hl(Vs, velocity_SL, velocity_M, q_of_liquid, q_of_gas)
			dencity_of_mixture = dencity_of_liquid * Hliquid + dencity_of_gas * (1 - Hliquid)
			grad_grav = calc_grad_grav(dencity_of_mixture)
			f = koef_of_friction(mode, velocity_SL, velocity_SG, inner_diameter, epsilon, Nd, dencity_of_liquid,
								 viscosity_of_liquid)
			grad_friction = calc_grad_fric(mode, f, velocity_SL, velocity_SG, velocity_M)
			grad_acceleration = calc_grad_acceleration(mode, velocity_M, velocity_SG, dencity_of_mixture,
													   current_pressure)

			grad_pressure_mode2 = calc_grad(mode, grad_friction, grad_grav, grad_acceleration)

			# Эмульсионный режим
			mode = 4
			S = define_S(mode, Nl, Nd, Nlv, Ngv)
			Vs = define_Vs(S, dencity_of_liquid, g, superficial_tencion_of_liquid)
			Hliquid = define_Hl(Vs, velocity_SL, velocity_M, q_of_liquid, q_of_gas)
			dencity_of_gas_corr = dencity_of_gas * Ngv / Ngv_Tr_or_M
			dencity_of_mixture = dencity_of_liquid * Hliquid + dencity_of_gas_corr * (1 - Hliquid)
			grad_grav = calc_grad_grav(dencity_of_mixture)
			f = koef_of_friction(mode, velocity_SL, velocity_SG, inner_diameter, epsilon, Nd, dencity_of_liquid,
								 viscosity_of_liquid)
			grad_friction = calc_grad_fric(mode, f, velocity_SL, velocity_SG, velocity_M)
			grad_acceleration = calc_grad_acceleration(mode, velocity_M, velocity_SG, dencity_of_mixture,
													   current_pressure)

			grad_pressure_mode4 = calc_grad(mode, grad_friction, grad_grav, grad_acceleration)

			# Расчет градиента давления для переходного режима
			A = (Ngv_Tr_or_M - Ngv) / (Ngv_Tr_or_M - Ngv_S_or_Tr)
			grad_pressure = A * grad_pressure_mode2 + (1 - A) * grad_pressure_mode4

		# Определение давления на текущей глубине
		if top_hole_pressure == 'None':
			current_pressure = bottom_hole_pressure - grad_pressure * current_depth
		else:
			current_pressure = top_hole_pressure + grad_pressure * current_depth

		depth_list.append(current_depth)
		pressure_list.append(current_pressure)

	return current_pressure, depth_list, pressure_list


def to_draw_graph_VLP(production, bottom_hole_pressure, length_tube, step_of_calc, THP = 'None'):
	pressure_list = []
	for i in production:
		q_of_liquid = i
		current_pressure, depth_this_iter, pressure_this_iter = calc_pressure(q_of_liquid, bottom_hole_pressure, length_tube, step_of_calc, THP)
		pressure_list.append(current_pressure)

	drawing(production, pressure_list, type_graph='VLP')


def toDrawGraphPressure(q_of_liquid, bottom_hole_pressure, length_tube, step_of_calc, THP = 'None'):
	current_pressure, depth_this_iter, pressure_this_iter = calc_pressure(q_of_liquid, bottom_hole_pressure, length_tube, step_of_calc, top_hole_pressure=THP)

	pressureList = np.array(pressure_this_iter)
	depthList_array = np.array(depth_this_iter)

	drawing(pressureList, depthList_array)





# Исходные данные ---------------------------------

# Плотность жидкости в поверхностных условиях, кг/м3
dencity_of_liquid = 762.64
# Плотность воды в поверхностных условиях, кг/м3
dencity_of_water = 1000
# Плотность газа в поверхностных условиях, кг/м3
dencity_of_gas = 20
# Удельная плотность газа в поверхностных условиях
relative_density_of_gas = 0.4

# Вязкость жидкости, Па*с
viscosity_of_liquid = 0.00097
# Вязкость газа, Па*с
viscosity_of_gas = 0.000016
g = 9.8
# Поверхностное натяжение жидкости
superficial_tencion_of_liquid = 0.00841

# Устьевая температура, град.К
well_head_temperature = 15 + 273
# Пластовая температура, град.К
strata_temperature = 100 + 273

# Длина НКТ, м
length_tube = 2000
# Диаметр НКТ, м
inner_diameter = 0.152
# Абсолютная шероховатость
epsilon = 0.000018288

# Площадь поперечного сечения, м2
Ap = 3.14 / 4 * inner_diameter ** 2
# Удельная плотность дегазрованной нефти
relative_density_of_oil = dencity_of_liquid / dencity_of_water
# Плотность нефти в API
specific_gravity_oil = 141.5 / relative_density_of_oil - 131.5
# -------------------------------------------------

# # Дебит жидкости в поверхностных условиях, м3/сут
# q_of_liquid_surface = 150
# Дебит газа в поверхностных условиях, тыс.м3/сут
q_of_gas_surface = 200
# # Устьевое давление (при расчете забойного), бар
# top_hole_pressure = 30
# Забойное давление (при расчете устьевого), бар
# bottom_hole_pressure = 100
# Шаг расчета, м
# step_of_calc = 20