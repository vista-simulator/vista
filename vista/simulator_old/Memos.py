class State:
    def __init__(self, translation_x=0.0, translation_y=0.0, theta=0.0):
        self.update(translation_x, translation_y, theta)

    def update(self, translation_x, translation_y, theta):
        self.translation_x = translation_x
        self.translation_y = translation_y
        self.theta = theta

    def reset(self):
        self.translation_x = 0.0
        self.translation_y = 0.0
        self.theta = 0.0
