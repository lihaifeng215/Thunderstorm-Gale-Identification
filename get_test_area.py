import numpy as np
'''
time = radar time
id = 0
x = x
y = y
wind = 0
'''

# 构造自动站数据
'''
get all AWS position
'''
def build_AWS_sample(Radar_image,Radar_time, radious):
    rows = Radar_image.shape[0] // (radious * 2 + 1)
    cols = Radar_image.shape[1] // (radious * 2 + 1)
    rows = [r * (radious * 2 + 1) + radious for r in range(rows - 1)]
    cols = [c * (radious * 2 + 1) + radious for c in range(cols - 1)]
    print("len(rows):", len(rows))
    print("len(cols):", len(cols))
    position = []
    for r in rows:
        for c in cols:
            position.append([r,c])
    position = np.array(position, dtype=int)
    print("position.shape:", position.shape)

    # 只是用有回波的position
    position_area = []
    area = (Radar_image > 20) & (Radar_image < 40)
    print("area.shape:",area.shape)
    for i in range(area.shape[0]):
        if area[int(position[i, 0]), int(position[i, 1])] == True:
            position_area.append(position[i, :])
    position = np.array(position_area, dtype=int)
    time = np.zeros(position.shape[0], dtype=int) + Radar_time
    id = np.zeros((position.shape[0],), dtype=int)
    wind = np.zeros((position.shape[0],), dtype=int)
    AWS_sample = np.column_stack((time, id, position, wind))
    return AWS_sample



if __name__ == '__main__':
    import time
    t = time.time()
    radar = np.random.random_integers(0, 80, (700,900))
    print("radar.shape:", radar.shape)
    AWS_sample = build_AWS_sample(radar,201803041012, radious=3)
    print(AWS_sample)
    print("AWS_sample.shape:", AWS_sample.shape)
    print(time.time() - t)


