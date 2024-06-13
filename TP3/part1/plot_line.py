import matplotlib.pyplot as plt


class Plot:
    def __init__(self, inputs, outputs, model) -> None:
        self.model = model
        fig = plt.figure()
        ax = fig.add_subplot()
        ax.set_xlim(-5, 5)
        ax.set_ylim(-5, 5)
        scatter = ax.scatter(*zip(*inputs), c=outputs, cmap="RdBu")
        fig.canvas.draw()

        self.axbackground = fig.canvas.copy_from_bbox(ax.bbox)

        (self.line,) = ax.plot([0, 0], [0, 0])
        self.fig, self.ax = fig, ax

        self.update()
        plt.show(block=False)

    def update(self, title=None):
        if title is not None:
            self.ax.set_title(title)
        self.fig.canvas.restore_region(self.axbackground)

        C, A, B = self.model.layers[0].weights[0]
        C = -C
        X = [-5, 5]
        Y = [-(A / B) * x - C / B for x in X]
        self.line.set_xdata(X)
        self.line.set_ydata(Y)
        self.ax.draw_artist(self.line)
        self.fig.canvas.blit(self.ax.bbox)
        self.fig.canvas.flush_events()
