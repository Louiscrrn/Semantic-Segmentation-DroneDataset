import plotly.express as px
from python.droneDataset import *

def plot_results(data, save_path=None, dump_name=None):
    """
    Plot training and validation metrics with bold axis labels and the ability to save the figures.

    Parameters:
        data (DataFrame): A pandas DataFrame containing 'epoch', 'train_loss', 'val_loss',
                          'train_iou', 'val_iou', 'train_accuracy', 'val_accuracy'.
        save_path (str): Path to save the figures.
        dump_name (str): Name for the saved file.
    """
    
    # Loss plot
    fig_loss = px.line(data, x='epoch', y=['train_loss', 'val_loss'], title="Training - Validation Loss with Epoch", template="ggplot2")
    fig_loss.update_layout(
        xaxis_title="<b>Epoch</b>",
        yaxis_title="<b>Loss</b>",
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="right",
            x=0.99
        )
    )
    fig_loss.show()
    
    # IoU plot
    fig_iou = px.line(data, x='epoch', y=['train_iou', 'val_iou'], title="Training - Validation IoU with Epoch", template="ggplot2")
    fig_iou.update_layout(
        xaxis_title="<b>Epoch</b>",
        yaxis_title="<b>IoU</b>",
        legend=dict(
            yanchor="bottom",
            y=0.01,
            xanchor="right",
            x=0.99
        )
    )
    fig_iou.show()
    
    # Accuracy plot
    fig_accuracy = px.line(data, x='epoch', y=['train_accuracy', 'val_accuracy'], title="Training - Validation Accuracy with Epoch", template="ggplot2")
    fig_accuracy.update_layout(
        xaxis_title="<b>Epoch</b>",
        yaxis_title="<b>Accuracy</b>",
        legend=dict(
            yanchor="bottom",
            y=0.01,
            xanchor="right",
            x=0.99
        )
    )
    fig_accuracy.show()
    
    fig_precision = px.line(data, x='epoch', y=['train_precision', 'val_precision'], title="Training - Validation Precision with Epoch", template="ggplot2")
    fig_precision.update_layout(
        xaxis_title="<b>Epoch</b>",
        yaxis_title="<b>Precision</b>",
        legend=dict(
            yanchor="bottom",
            y=0.01,
            xanchor="right",
            x=0.99
        )
    )
    fig_precision.show()
    
    # Save the plots if save_path is specified
    if save_path and dump_name:
        fig_loss.write_image(save_path + dump_name + "_loss.png", width=800, height=400,scale=2)
        fig_iou.write_image( save_path + dump_name + "_iou.png", width=800, height=400,scale=2)
        fig_accuracy.write_image(save_path + dump_name + "_accuracy.png", width=800, height=400,scale=2)
        fig_precision.write_image(save_path + dump_name + "_precision.png", width=800, height=400,scale=2)

def get_axes_mask(mask, cmap=COLOR_MAP, axes=None):
 
    mask_rgb = class_to_rgb(mask, cmap)
    
    mask_rgb_np = mask_rgb.cpu().numpy()

    colors = np.array([c[0] for c in cmap.values()]) / 255.0
    labels = [c[1] for c in cmap.values()]

    if axes is None:
        _, axes = plt.subplots(figsize=(6, 6))

    axes.imshow(mask_rgb_np)
    axes.axis("off")

    legend_elements = [
        Patch(facecolor=color, label=label) for color, label in zip(colors, labels)
    ]
    axes.legend(handles=legend_elements, loc='best', title="Classes")
    
    return axes
