class ValueTracker:
    def __init__(self, alpha=0.0):
        """ Track a value (or array) with potential temporal smoothening

        Inputs:
            alpha (float): exponential smoothing factor
        """
        self.alpha = alpha
        self.x = None

    def update(self, new):
        """ Update the state of the main variable and apply smoothening factor

        Inputs:
            new (float or np.ndarray): new value to add to the tracker
        """
        if self.x is None:
            self.x = new
        else:
            self.x = (1-self.alpha)*new + (self.alpha)*self.x

    def reset(self):
        """ Reset the state of the tracker """
        self.x = None

    def get(self):
        """ Get the numerical state """
        return self.x

    def __repr__(self):
        return repr(self.x)
