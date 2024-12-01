import matplotlib.pyplot as plt

def plot_values(epochs_seen, examples_seen, train_values, val_values, label="loss"):
    """
    Plot training and validation values (e.g., loss or accuracy) over epochs and examples seen.

    Args:
        epochs_seen: Tensor or list of epoch numbers corresponding to the training values.
        examples_seen: Tensor or list of examples seen corresponding to the training values.
        train_values: List of training values (e.g., loss or accuracy).
        val_values: List of validation values (e.g., loss or accuracy).
        label: Label for the values being plotted (e.g., "loss", "accuracy").

    Saves:
        A PDF file of the plot with the name `{label}-plot.pdf`.
    """
    fig, ax1 = plt.subplots(figsize=(5, 3))

    # Plot training and validation values against epochs
    ax1.plot(epochs_seen, train_values, label=f"Training {label}")
    ax1.plot(epochs_seen, val_values, linestyle="-.", label=f"Validation {label}")
    ax1.set_xlabel("Epochs")
    ax1.set_ylabel(label.capitalize())
    ax1.legend()

    # Create a second x-axis for examples seen
    ax2 = ax1.twiny()  # Create a second x-axis that shares the same y-axis
    ax2.plot(examples_seen, train_values, alpha=0)  # Invisible plot for aligning ticks
    ax2.set_xlabel("Examples seen")

    fig.tight_layout()  # Adjust layout to make room
    plt.savefig(f"{label}-plot.pdf")
    plt.show()
