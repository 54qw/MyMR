import numpy as np

def objectiveFunction1(C, X):
    total_distance = 0
    for i, tour in enumerate(X):
        distance = np.sum(np.multiply(C, tour))
        total_distance += distance
    return (-total_distance)



def objectiveFunction2(C, T, X, ptype):
    n_tours = X.shape[0]
    tour_lengths = np.empty(shape=(n_tours,), dtype=float)
    tour_times = np.empty(shape=(n_tours,), dtype=float)
    total_distance = 0
    total_time = 0
    avg_tour_length = 0
    variation = 0

    if ptype == 2:
        for i, tour in enumerate(X):
            tourtime = np.sum(np.multiply(T, tour))
            total_time += tourtime
            tour_times[i] = tourtime
        avg_tour_time = total_time / n_tours
        for i in range(n_tours):
            variation += np.abs(avg_tour_time - total_time[i])
        return (-variation) * 60

    if ptype == 1:
        for i, tour in enumerate(X):
            tour_lengths[i] = np.sum(np.multiply(C, tour))
        return -(max(tour_lengths) - min(tour_lengths))



