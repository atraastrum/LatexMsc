import numpy as np

def generate(r, d, w, numa, numb):
    a = np.empty((numa, 2))
    # Generating points for region A
    small_radius = r - w/2.
    big_radius = small_radius + w
    while numa > 0:
        x = np.random.uniform(-big_radius, big_radius)
        y = np.random.uniform(0, big_radius)

        dist = x**2 + y**2
        if small_radius**2 < dist < big_radius**2:
            numa = numa - 1
            a[numa, 0] = x
            a[numa, 1] = y

    b = np.empty((numb, 2))
    # Generating point for B
    while numb > 0:
        x = np.random.uniform(-w/2., 2. * big_radius - w/2.)
        y = np.random.uniform(-d, -d - big_radius)
        dist = (x-r)**2 + (y + d)**2
        if small_radius**2 < dist < big_radius**2:
            numb = numb - 1
            b[numb, 0] = x
            b[numb, 1] = y

    return a, b

numa = 500
numb = numa
(a_data, b_data) = generate(r=10, d=-4, w=6, numa=numa, numb=numb)
X = np.vstack((a_data, b_data))
y = np.hstack((np.ones(numa), -1 * np.ones(numb))).reshape(numa+numb, 1)
