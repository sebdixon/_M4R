import numpy as np
import pandas as pd
import plotly.express as px


def load_df(filename: str) -> pd.DataFrame:
    return pd.read_csv(filename)

def plot_metrics(
        df: pd.DataFrame, 
        metric: str, 
        title: str,
        save_path: str = None
) -> None:
    fig = px.line(
        df,
        x='Samples',
        y='Score',
        color='Method',
        title='title',
        labels={'Score': metric, 'Samples': 'Number of simulations'}
    )
    # get rid of gridlines
    fig.update_xaxes(
        showgrid=False,
        tickvals=[1, 5, 10],
        ticktext=['1k', '5k', '10k'])
    fig.update_layout(
        title=title,
        xaxis_title='Number of simulations',
        yaxis_title=metric,
        paper_bgcolor='white',
        plot_bgcolor='white'
    )
    fig.update_traces(mode='markers+lines')
    if metric == 'c2st':
        fig.update_yaxes(range=[0.5, 1])
    elif metric == 'meddist':
        baseline_meddist = 97.76758123811723
        fig.add_shape(
            type='line',
            x0=1,
            y0=baseline_meddist,
            x1=10,
            y1=baseline_meddist,
            line=dict(
                color='black',
                width=2,
                dash='dashdot'
            )
        )
    if save_path is not None:
        fig.write_image(save_path)
    fig.show()


if __name__ == "__main__":
    df = load_df('simulated_data/power_law/c2st_scores.csv')
    plot_metrics(
        df=df, 
        metric='c2st', 
        title='C2ST scores for power law model',
        save_path='simulated_data/power_law/plots/c2st_scores.png')
    df = load_df('simulated_data/power_law/meddist_scores.csv')
    plot_metrics(
        df=df, 
        metric='meddist', 
        title='PPC median distance for power law model',
        save_path='simulated_data/power_law/plots/meddist_scores.png')
    df = load_df('simulated_data/power_law/sequential/c2st_scores_SNPE.csv')
    plot_metrics(
        df=df, 
        metric='c2st', 
        title='Sequential C2ST scores for power law model',
        save_path='simulated_data/power_law/plots/sequential/c2st_scores_SNPE.png')