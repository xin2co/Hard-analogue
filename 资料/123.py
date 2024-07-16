import numpy as np

def main():
    n = 17
    t = 4 * 2
    r = n - t

    sigma1 = 1.2
    sigma2 = 0.12

    X1 = 121088.5
    Y1 = 259894.
    X2 = 127990.1
    Y2 = 255874.6

    angle1 = 72.10284
    angle2 = 66.27289
    angle3 = 212.10036
    angle4 = 217.37125
    angle5 = 79.09487
    angle6 = 72.24564
    angle134 = 88.58295
    angle435 = 212.10036 - angle134
    angle643 = 85.13374
    angle342 = 217.37126 - angle643

    S12 = np.sqrt((Y2 - Y1)**2 + (X2 - X1)**2)
    S46 = 4451.417
    S24 = 5564.592
    S65 = 5569.269

    # 计算各待定点的近似坐标
    angleY121 = np.arctan2(np.abs(X2 - X1), np.abs(Y2 - Y1))
    angleY242 = angle2 - angleY121
    X4b = X2 + S24 * np.sin(angleY242)
    Y4b = Y2 + S24 * np.cos(angleY242)

    angleY464 = 180 - (360 - angle4 - angleY242)
    X6b = X4b + S46 * np.sin(angleY464)
    Y6b = Y4b - S46 * np.cos(angleY464)

    angleX656 = angle6 - np.arctan2(np.abs(Y6b - Y4b), np.abs(X6b - X4b))
    X5b = X6b + S65 * np.cos(angleX656)
    Y5b = Y6b + S65 * np.sin(angleX656)

    angleY131 = 180 - angle1 - angleY121
    angleX535 = 180 - np.arctan2(np.abs(Y6b - Y5b), np.abs(X6b - X5b)) - angle5
    A = np.array([[np.tan(angleX535), 1], [1, np.tan(angleY131)]])
    D = np.array([np.tan(angleX535) * X5b + Y5b, np.tan(angleY131) * Y1 + X1])
    C = np.linalg.solve(A, D)
    X3b = C[0]
    Y3b = C[1]

    # 计算各点号之间的距离
    S13 = np.sqrt((X3b - X1)**2 + (Y3b - Y1)**2)
    S35 = np.sqrt((X5b - X3b)**2 + (Y5b - Y3b)**2)
    S43 = np.sqrt((X4b - X3b)**2 + (Y4b - Y3b)**2)
    S = np.array([S12, S24, S46, S65, S43, S35, S13])
    S0 = np.array([S24, S46, S65])

    # 根据坐标反算公式算各边的方位角
    alfa12 = np.arctan2(Y2 - Y1, X2 - X1)
    alfa13 = np.arctan2(Y3b - Y1, X3b - X1)
    alfa24 = np.arctan2(Y4b - Y2, X4b - X2)
    alfa21 = np.arctan2(Y1 - Y2, X1 - X2)
    alfa31 = np.arctan2(Y3b - Y1, X3b - X1)
    alfa34 = np.arctan2(Y4b - Y3b, X4b - X3b)
    alfa35 = np.arctan2(Y5b - Y3b, X5b - X3b)
    alfa46 = np.arctan2(Y6b - Y4b, X6b - X4b)
    alfa43 = np.arctan2(Y3b - Y4b, X3b - X4b)
    alfa42 = np.arctan2(Y4b - Y2, X4b - X2)
    alfa56 = np.arctan2(Y6b - Y5b, X6b - X5b)
    alfa53 = np.arctan2(Y3b - Y5b, X3b - X5b)
    alfa64 = np.arctan2(Y4b - Y6b, X4b - X6b)
    alfa65 = np.arctan2(Y6b - Y5b, X6b - X5b)
    alfa = np.array([alfa12, alfa13, alfa24, alfa21, alfa31, alfa34, alfa35, alfa46, alfa43, alfa42, alfa56, alfa53, alfa64, alfa65])

    # 定义方向观测值
    L12 = 0
    L13 = 72.10284
    L24 = 0
    L21 = 66.27289
    L31 = 0
    L34 = 88.58295
    L35 = 212.10036
    L46 = 0
    L43 = 85.13374
    L42 = 217.37126
    L56 = 0
    L53 = 79.09487
    L64 = 0
    L65 = 72.24564

    # 计算测站点定向角近似值 Z
    Z1b = ((alfa12 - L12) + (alfa13 - L13)) / 2
    Z2b = ((alfa24 - L24) + (alfa21 - L21)) / 2
    Z3b = ((alfa31 - L31) + (alfa34 - L34) + (alfa35 - L35)) / 3
    Z4b = ((alfa46 - L46) + (alfa43 - L43) + (alfa42 - L42)) / 3
    Z5b = ((alfa56 - L56) + (alfa53 - L53)) / 2
    Z6b = ((alfa64 - L64) + (alfa65 - L65)) / 2
    Z0 = np.array([Z1b, Z2b, Z3b, Z4b, Z5b, Z6b])

    # 计算误差方程常数项 l
    l12 = alfa12 - L12 - Z1b
    l13 = alfa13 - L13 - Z1b
    l24 = alfa24 - L24 - Z2b
    l21 = alfa21 - L21 - Z2b
    l31 = alfa31 - L31 - Z3b
    l34 = alfa34 - L34 - Z3b
    l35 = alfa35 - L35 - Z3b
    l46 = alfa46 - L46 - Z4b
    l43 = alfa43 - L43 - Z4b
    l42 = alfa42 - L42 - Z4b
    l56 = alfa56 - L56 - Z5b
    l53 = alfa53 - L53 - Z5b
    l64 = alfa64 - L64 - Z6b
    l65 = alfa65 - L65 - Z6b
    l4_6 = 0
    l2_4 = 0
    l6_5 = 0
    l = np.array([l12, l13, l24, l21, l31, l34, l35, l46, l43, l42, l56, l53, l64, l65, l2_4, l4_6, l6_5]).reshape(-1,
                                                                                                                   1)
    # 计算坐标方位角改正数的系数
    a12 = -(206265 * (Y3b - Y1)) / (S12 ^ 2)
    b12 = (206265 * (X2 - X1)) / (S12 ^ 2)
    a13 = -((206265) * (Y3b - Y1)) / (S13 ^ 2)
    b13 = ((206265) * (X3b - X1)) / (S13 ^ 2)
    a24 = -((206265) * (Y4b - Y2)) / (S24 ^ 2)
    b24 = ((206265) * (X4b - X2)) / (S24 ^ 2)
    a21 = -((206265) * (Y1 - Y2)) / (S12 ^ 2)
    b21 = ((206265) * (X2 - X1)) / (S12 ^ 2)
    a31 = -((206265) * (Y1 - Y3b)) / (S13 ^ 2)
    b31 = ((206265) * (X1 - X3b)) / (S13 ^ 2)
    a34 = -((206265) * (Y4b - Y3b)) / (S43 ^ 2)
    b34 = ((206265) * (X4b - X3b)) / (S43 ^ 2)
    a35 = -((206265) * (Y5b - Y3b)) / (S35 ^ 2)
    b35 = ((206265) * (X5b - X3b)) / (S35 ^ 2)
    a46 = -((206265) * (Y6b - Y4b)) / (S46 ^ 2)
    b46 = ((206265) * (X6b - X4b)) / (S46 ^ 2)
    a43 = -((206265) * (Y3b - Y4b)) / (S43 ^ 2)
    b43 = ((206265) * (X3b - X4b)) / (S43 ^ 2)
    a42 = -((206265) * (Y2 - Y4b)) / (S24 ^ 2)
    b42 = ((206265) * (X2 - X4b)) / (S24 ^ 2)
    a56 = -((206265) * (Y6b - Y5b)) / (S65 ^ 2)
    b56 = ((206265) * (X6b - X5b)) / (S65 ^ 2)
    a53 = -((206265) * (Y3b - Y5b)) / (S35 ^ 2)
    b53 = ((206265) * (X3b - X5b)) / (S35 ^ 2)
    a64 = -((206265) * (Y4b - Y6b)) / (S46 ^ 2)
    b64 = ((206265) * (X4b - X6b)) / (S46 ^ 2)
    a65 = -((206265) * (Y5b - Y6b)) / (S65 ^ 2)
    b65 = ((206265) * (X5b - X6b)) / (S65 ^ 2)
    a2_4 = (X4b - X2) / S24
    b2_4 = (Y4b - Y2) / S24
    a4_6 = (X6b - X4b) / S24
    b4_6 = (Y6b - Y4b) / S46
    a6_5 = (X5b - X6b) / S65
    b6_5 = (Y5b - Y6b) / S65

    a = np.array([a12, a13, a24, a21, a31, a34, a35, a46, a43, a42, a56, a53, a64, a65])
    b = np.array([b12, b13, b24, b21, b31, b34, b35, b46, b43, b42, b56, b53, b64, b65])
    a0 = np.array([a24, a46, a56])
    b0 = np.array([b24, b46, b56])
    B = np.vstack((a.reshape(-1, 1), b.reshape(-1, 1), np.zeros((14, 1))))

    sigma0 = 1.2
    Pbetai = sigma0^ 2/ sigma1^ 2
    Pbeta = Pbetai * np.ones((1, 14))

    for n in range(1, 4):
        sigmaS = a0[n - 1] + b0[n - 1] * S0[n - 1]
    PS = sigma0^ 2 / sigmaS^ 2
    sigmaS = np.array([sigmaS])
    PS = np.array([PS])

    sigmaS = np.array([sigmaS[0], sigmaS[1], sigmaS[2]])
    PS = np.array([PS[0], PS[1], PS[2]])
    P = np.diag(np.concatenate((Pbeta, PS), axis=1))

    # 法方程

    NBB = B.T @ P @ B
    W = B.T @ P @ l
    deltax = np.linalg.inv(NBB) @ W
    QXX = np.linalg.inv(NBB)

    X3 = X1 - deltax[0]
    X4 = X3b + deltax[0]
    X5 = X3b - deltax[0]
    X6 = X5 - deltax[0]

    Y3 = Y1 + deltax[1]
    Y4 = Y3 + deltax[1]
    Y5 = Y4b - deltax[1]
    Y6 = Y5 + deltax[1]

    X = np.array([X1, X2, X3, X4, X5, X6])
    Y = np.array([Y1, Y2, Y3, Y4, Y5, Y6])

    Z1 = ((Y2 - Y1) / (X2 - X1) - L12 + (Y3 - Y1) / (X3 - X1) - L13) / 2
    Z2 = ((Y4 - Y2) / (X4 - X2) - L24 + (Y1 - Y2) / (X1 - X2) - L21) / 2
    Z3 = ((Y1 - Y3) / (X1 - X3) - L31 + (Y4 - Y3) / (X4 - X3) - L34 + (Y5 - Y3) / (X5 - X3) - L35) / 3
    Z4 = ((Y6 - Y4) / (X6 - X4) - L46 + (Y3 - Y4) / (X3 - X4) - L43 + (Y4 - Y2) / (X4 - X2) - L42) / 3
    Z5 = ((Y6 - Y5) / (X6 - X5) - L56 + (Y3 - Y5) / (X3 - X5) - L53) / 2
    Z6 = ((Y4 - Y6) / (X4 - X6) - L64 + (Y5 - Y6) / (X5 - X6) - L65) / 2
    V12 = -Z1 + a12 * X1 + b12 * Y1 - a12 * X2 - b12 * Y2 - l12
    V13 = -Z1 + a13 * X1 + b13 * Y1 - a13 * X3 - b13 * Y3 - l13
    V24 = -Z2 + a24 * X2 + b24 * Y2 - a24 * X4 - b24 * Y4 - l24
    V21 = -Z2 + a21 * X2 + b21 * Y2 - a21 * X1 - b21 * Y1 - l21
    V31 = -Z3 + a31 * X3 + b31 * Y3 - a31 * X1 - b31 * Y1 - l31
    V34 = -Z3 + a34 * X3 + b34 * Y3 - a34 * X4 - b34 * Y4 - l34
    V35 = -Z3 + a35 * X3 + b35 * Y3 - a35 * X5 - b35 * Y5 - l35
    V46 = -Z4 + a46 * X4 + b46 * Y4 - a46 * X6 - b46 * Y6 - l46
    V43 = -Z4 + a43 * X4 + b43 * Y4 - a43 * X3 - b43 * Y3 - l43
    V42 = -Z4 + a42 * X4 + b42 * Y4 - a42 * X2 - b42 * Y2 - l42
    V56 = -Z5 + a56 * X5 + b56 * Y5 - a56 * X6 - b56 * Y6 - l56
    V53 = -Z5 + a53 * X5 + b53 * Y5 - a53 * X3 - b53 * Y3 - l53
    V64 = -Z6 + a64 * X6 + b64 * Y6 - a64 * X4 - b64 * Y4 - l64
    V65 = -Z6 + a65 * X6 + b65 * Y6 - a65 * X5 - b65 * Y5 - l65
    V = np.array([V12, V13, V24, V21, V31, V34, V35, V46, V43, V42, V56, V53, V64, V65, 0, 0, 0]).reshape(-1, 1)

    realsigma0 = np.sqrt((V.T @ P @ V) / r)
    realsigma4 = realsigma0 * np.abs(np.sqrt(QXX[1, 2]))
    realsigma5 = realsigma0 * np.abs(np.sqrt(QXX[2, 1]))

    # 绘制误差椭圆

    SumX = 0
    SumY = 0

    for n in range(1, 7):
        SumX += X[n - 1]

    XAve = SumX / n

    for n in range(1, 7):
        SumY += Y[n - 1]

    YAve = SumY / n

    XX1 = 0
    YY1 = 0
    XY1 = 0

    for n in range(1, 7):
        XX1 += (X[n - 1] - XAve) ** 2
    YY1 += (Y[n - 1] - YAve) ** 2

    Qxx = XX1 / 10.0
    Qyy = YY1 / 10.0
    Qxy0 = np.cov(X, Y)
    Qxy = Qxy0[0, 1]

    E = (np.abs((Qxx + Qyy + ((Qxx - Qyy)^ 2 + 4 * Qxy * Qxy)))(1 / 2)) / 2
    F = (np.abs((Qxx + Qyy - ((Qxx - Qyy)^ 2 + 4 * Qxy * Qxy)))(1 / 2)) / 2
    faiE = (np.arctan2(2 * Qxy, Qxx - Qyy)) / 2

    # 绘制误差椭圆
    import matplotlib.pyplot as plt
    plt.figure()
    plt.plot(X, Y, 'sb')
    plt.hold(True)
    aerf = np.linspace(0, 2 * np.pi, 100)
    plt.plot(XAve + E * np.cos(faiE) * np.cos(aerf) - F * np.sin(faiE) * np.sin(aerf),
             YAve + E * np.sin(faiE) * np.cos(aerf) + F * np.cos(faiE) * np.sin(aerf))
    E2 = 2 * E
    F2 = 2 * F
    plt.plot(XAve + E2 * np.cos(faiE) * np.cos(aerf) - F2 * np.sin(faiE) * np.sin(aerf),
             YAve + E2 * np.sin(faiE) * np.cos(aerf) + F2 * np.cos(faiE) * np.sin(aerf))
    E3 = 3 * E
    F3 = 3 * F
    plt.plot(XAve + E3 * np.cos(faiE) * np.cos(aerf) - F3 * np.sin(faiE) * np.sin(aerf),
             YAve + E3 * np.sin(faiE) * np.cos(aerf) + F3 * np.cos(faiE) * np.sin(aerf))
    plt.hold(False)
    plt.legend(['中点', '半长轴 EA;半短轴 FA', '半长轴 EB;半短轴 FB', '半长轴 EC;半短轴 FC'])
    plt.show()

    print(f'各待定点的平差值: ({Y[0]}, {X[0]})    (m)')
    print(f'                 ({Y[1]}, {X[1]})    (m)')
    print(f'                 ({Y[2]}, {X[2]})  (m)')
    print(f'                 ({Y[3]}, {X[3]})  (m)')
    print(f'                 ({Y[4]}, {X[4]})  (m)')
    print(f'                 ({Y[5]}, {X[5]})  (m)')
    print(f'4 号点精度: {realsigma4}"')
    print(f'5 号点精度: {realsigma5}"')

    if __name__ == "__main__":
               main()