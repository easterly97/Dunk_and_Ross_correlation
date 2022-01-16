import math


# Вязкость газа
def toDefineViscosityGasRash(densityGas, tempKel, moleKmGas):
    """
    Функция, рассчитывающая вязкость по корреляции Ли. Работает при условии отсутствия сернистых соединений
    :param densityGas: Плотность газа, [кг/м3]
    :param tempKel: Температура, [град.К]
    :param moleKmGas: Молекулярный вес газовой смеси, [г/моль]
    :return: Значение динамической вязкости, [Па/с]
    """
    K = (9.4 + 0.02 * moleKmGas) * (1.8 * tempKel) ** 1.5 / (209 + 19 * moleKmGas + 1.8 * tempKel)
    X = 3.5 + (547.8 / tempKel) + 0.01 * moleKmGas
    Y = 2.4 - 0.2 * X

    return 0.0000001 * K * math.exp(X * (densityGas / 1000) ** Y)


# Коэф. сверхсжимаемости
def toDefineKoefZ(currentPressure, relDensityGas, currentTemp):
    """
    Функция для расчета коэффициента сверхсжимаемости по корреляции Беггса и Брила при определенных термобарических условиях
    :param currentPressure: Текущее давление, [бар]
    :param relDensityGas: Относительная плотность газа
    :param currentTemp: Текущая температура, [град.К]
    :return: Значение коэффициента сверхсжимаемости
    """

    if currentPressure < 0.101325:
        currentPressure = 0.101325

    # Расчет псевдокритических параметров (Давление в барах)
    # Природный газ
    if relDensityGas <= 0.575:
        pressurePseudoCrit = 0.0689 * (677 + 15 * relDensityGas - 37.5 * relDensityGas ** 2)
        temperaturePseudoCrit = (168 + 325 * relDensityGas - 12.5 * relDensityGas ** 2) / 1.8
    # Газоконденсатный газ
    else:
        pressurePseudoCrit = 0.0689 * (706 + 51.7 * relDensityGas - 11.1 * relDensityGas ** 2)
        temperaturePseudoCrit = (187 + 330 * relDensityGas - 71.5 * relDensityGas ** 2) / 1.8

    # Приведенные давление и температура
    pressurePr = currentPressure / pressurePseudoCrit
    temperaturePr = currentTemp / temperaturePseudoCrit

    A = 1.39 * (temperaturePr - 0.92) ** 0.5 - 0.36 * temperaturePr - 0.101
    E = 9 * (temperaturePr - 1)
    B = (0.62 - 0.23 * temperaturePr) * pressurePr + (
            0.066 / (temperaturePr - 0.86) - 0.037) * pressurePr ** 2 + 0.32 * pressurePr ** 6 / 10 ** E
    C = 0.132 - 0.32 * math.log10(temperaturePr)
    F = 0.3106 - 0.49 * temperaturePr + 0.1824 * temperaturePr ** 2
    D = 10 ** F

    # Коэф. сверхсжимаемости
    Z = A + (1 - A) / math.exp(B) + C * pressurePr ** D

    return Z


# Плотность газа
def toDefineCurrentDensityGas(relDensity, pressure, temperature, koefZ):
    """
    Расчет абсолютной плотности газа для конкретных термобарических условий.
    :param relDensity: Относительная плотность газа
    :param pressure: Давление газа, [Па]
    :param temperature: Абсолютная температура газа, [град.К]
    :param koefZ: Коэффициент сверхсжимаемости
    :return: Абсолютная плотность газа, [кг/м3]
    """
    return 3.5 * 0.001 * relDensity * pressure / temperature / koefZ

