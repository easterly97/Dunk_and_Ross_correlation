import matplotlib.pyplot as plt

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