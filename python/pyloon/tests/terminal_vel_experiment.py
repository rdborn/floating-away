for i in range(100):
    u = i / 10.0
    v = 0.0
    rho = 0.25
    Cd = 0.5
    A = 10.0
    C = rho * Cd * A / 2.0
    dt = 0.1
    w = u - v
    threshold = 1e-3
    T = 0
    w10 = w*0.1
    w90 = w*0.9
    while w > threshold:
        a = C * w * abs(w)
        v = v + a * dt
        if w < w90 and w > w10:
            T += dt
        w = u - v
    print("u: " + str(u) + ", T: " + str(T))
