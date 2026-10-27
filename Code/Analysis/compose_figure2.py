"""Arrange the existing Figure 2a–c panels with the recalculated panels d–e."""
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.image as mpimg
import matplotlib.pyplot as plt

PACKAGE = Path(__file__).resolve().parents[2]
FIGURES = PACKAGE / "Figures/Main_Figures"

def main():
    top = mpimg.imread(FIGURES / "Figure2_abc.png")
    bottom = mpimg.imread(FIGURES / "Figure2_de.png")
    top_height = top.shape[0] / top.shape[1]
    bottom_height = bottom.shape[0] / bottom.shape[1]
    total = top_height + bottom_height
    fig = plt.figure(figsize=(17.5/2.54, 17.5*total/2.54), dpi=350, facecolor="white")
    ax_top = fig.add_axes([0, bottom_height/total, 1, top_height/total])
    ax_bottom = fig.add_axes([0, 0, 1, bottom_height/total])
    for ax, bitmap in ((ax_top, top), (ax_bottom, bottom)):
        ax.imshow(bitmap)
        ax.set_axis_off()
    fig.savefig(FIGURES / "Figure2.png", dpi=350, facecolor="white", pad_inches=0)
    plt.close(fig)

if __name__ == "__main__":
    main()
