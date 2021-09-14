import numpy as np


def sample_vogel_disc(centre, radius, num_samples):
    phi = (1 + np.sqrt(5)) / 2

    samples = []
    for i in range(1, num_samples + 1):
        distance = radius * np.sqrt(i) / np.sqrt(num_samples + 1)
        angle = 2 * np.pi * phi * i

        x = distance * np.cos(angle) + centre[0]
        y = distance * np.sin(angle) + centre[1]
        samples.append(np.array([x, y, centre[2]]))
    return samples