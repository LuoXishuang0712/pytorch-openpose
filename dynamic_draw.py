import matplotlib.pyplot as plt
import numpy as np

class dynamic_draw:
    def __init__(self, height, width) -> None:
        plt.ion()
        self.fig = plt.figure()
        self.height = height
        self.width = width
    
    def build_lines(self, motion):
        if motion.shape[0] == 1:
            motion = motion[0]
        line = []
        body_line = [
            (0, 1), (0, 15), (0, 16), (1, 2), (1, 5), (1, 8), (2, 3), (3, 4), (5, 6), (6, 7),
            (8, 9), (8, 12), (9, 10), (10, 11), (11, 22), (11, 24), (12, 13), (13, 14),
            (14, 19), (14, 21), (15, 17), (16, 18), (19, 20), (22, 23)
        ]
        for i in range(len(body_line)):
            item = body_line[i]
            if (motion[item[0]][2] == 0 or motion[item[1]][2] == 0):
                continue
            line.append([
                [motion[item[0]][0], motion[item[1]][0]],
                [motion[item[0]][1], motion[item[1]][1]]
            ])
        return line

    def motion_update(self, motion_data, image=None):
        plt.cla()

        x = motion_data.T[0]
        y = motion_data.T[1]

        plt.scatter(x=x, y=y, c='r', s=70)
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.xlim((0, self.width))
        plt.ylim((0, self.height))

        if image is not None:
            plt.imshow(image)

        lines = self.build_lines(motion_data)
        for line in lines:
            plt.plot(line[0], line[1])
        
        plt.draw()
        plt.show()
        plt.pause(0.01)
