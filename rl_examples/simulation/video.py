"""Stream MP4 frames; save the last frame as a README preview image."""

from pathlib import Path
import imageio.v2 as imageio
import numpy as np
import matplotlib.pyplot as plt
from rl_examples.configs.settings import NavigationConfig
from rl_examples.utils.labels import condition_label


class VideoRecorder:
    def __init__(self, env, config, path):
        self.env, self.config, self.path = env, config, Path(path)
        self.navigation = isinstance(config, NavigationConfig)
        self.fps = 2 if self.navigation else 50
        self.figure = None
        self.last_frame = None
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.writer = imageio.get_writer(
            str(self.path), fps=self.fps, codec="libx264", macro_block_size=1
        )

    def capture(self, step, outcome="running"):
        if self.navigation:
            from rl_examples.visualization.plots import plot_navigation

            if self.figure is None:
                self.figure, self.ax = plt.subplots(figsize=(6, 6), dpi=100)
                plot_navigation(self.env, ax=self.ax)
                self.figure.tight_layout()
            path = np.asarray(self.env.path)
            self.ax.lines[0].set_data(path[:, 0], path[:, 1])
            self.ax.patches[-1].center = self.env.robot.pose[:2]
            self.ax.set_title(
                f"{condition_label(self.config)} | decision = {step}\n{outcome}"
            )
            self.figure.canvas.draw()
            frame = np.asarray(self.figure.canvas.buffer_rgba())[:, :, :3].copy()
        else:
            frame = self.env.render()
        self.writer.append_data(frame)
        self.last_frame = frame

    def close(self):
        try:
            self.writer.close()
            if self.last_frame is not None:
                imageio.imwrite(self.path.with_suffix(".png"), self.last_frame)
        finally:
            if self.figure is not None:
                plt.close(self.figure)
