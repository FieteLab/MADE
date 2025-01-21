import numpy as np
import matplotlib.pyplot as plt

TOPOLOGIES = ["Plane", "Cylinder", "Torus", "Sphere", "MobiusBand"]


def main():
    for topology in TOPOLOGIES:
        # load embeddings
        embeddings = np.load(f"./imgs/{topology}_isomap.npy")

        # plot embeddings
        f = plt.figure()
        ax = f.add_subplot(111, projection="3d")
        ax.scatter(
            embeddings[:, 0],
            embeddings[:, 1],
            embeddings[:, 2],
            c=embeddings[:, 2],
            s=5,
        )

        # remove grid
        ax.grid(False)
        ax.set(
            xticks=[],
            yticks=[],
            zticks=[],
        )

        plt.savefig(f"./imgs/{topology}_isomap.png")


if __name__ == "__main__":
    main()
