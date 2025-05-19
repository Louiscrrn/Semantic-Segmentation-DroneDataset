import yaml
import pandas as pd
import plotly.express as px
import os

def get_plot(data, type) :
    fig = px.line(data, x='epoch', y=[f'train_{type}', f'val_{type}'], title=f"Training - Validation {type} with Epoch", template="ggplot2")
    fig.update_layout(
        xaxis_title="<b>Epoch</b>",
        yaxis_title=f"<b>{type}</b>",
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="right",
            x=0.99
        ) if type == 'loss' else dict(yanchor="bottom",y=0.01,xanchor="right",x=0.99
        )
    )
    return fig

def save_fig(save_path, dump_name, fig_loss, fig_iou, fig_iou_w, fig_mpa, fig_mpa_w) :
    if save_path and dump_name:
        fig_loss.write_image(save_path + dump_name + "_loss.png", width=800, height=400,scale=2)
        fig_iou.write_image( save_path + dump_name + "_iou.png", width=800, height=400,scale=2)
        fig_iou_w.write_image( save_path + dump_name + "_iou_w.png", width=800, height=400,scale=2)
        fig_mpa.write_image(save_path + dump_name + "_mpa.png", width=800, height=400,scale=2)
        fig_mpa_w.write_image(save_path + dump_name + "_mpa_w.png", width=800, height=400,scale=2)
        return 0
    else :
        return -1


def main(config_path="config.yaml") :
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    save_path = config['training']['output_path']
    dump_name = config['model']['type']

    data = pd.read_csv( save_path + '/' + dump_name + '/' + config['training']['train_hist_name'])
    
    fig_loss = get_plot(data, "loss")
    fig_loss.show()
    
    fig_iou = get_plot(data, "iou")
    fig_iou.show()
    
    fig_iou_w = get_plot(data, "iou_w")
    fig_iou_w.show()
    
    fig_mpa = get_plot(data, "mpa")
    fig_mpa.show()
    
    fig_mpa_w = get_plot(data, "mpa_w")
    fig_mpa_w.show() 
    
    return save_fig(save_path, dump_name, fig_loss, fig_iou, fig_iou_w, fig_mpa, fig_mpa_w)
    

if __name__ == "__main__":
    if 0 == main() :
        print(f'Figures successfully saved !') 
    else :
        print(f"Erreur de sauvegarde !")
