import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from plotly.subplots import make_subplots
from scipy.stats import binom


def load_df(filename: str) -> pd.DataFrame:
    return pd.read_csv(filename)

def plot_metrics(
        df: pd.DataFrame, 
        metric: str, 
        title: str,
        save_path: str = None,
        logscale: bool = False,
        tickvals: list[int] = [1, 5, 10],
        ticktext: list[str] = [r'$1k$', r'$5k$', r'$10k$']
) -> None:
    fig = px.line(
        df,
        x='Samples',
        y='Score',
        color='Method',
        title='title',
        labels={'Score': metric, 'Samples': r'$Number of simulations$'}
    )
    # get rid of gridlines
    fig.update_xaxes(
        showgrid=False,
        tickvals=tickvals,
        ticktext=ticktext)
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
        if logscale:
            fig.add_shape(
                type='line',
                x0=1,
                y0=baseline_meddist,
                x1=100,
                y1=baseline_meddist,
                line=dict(
                    color='black',
                    width=2,
                    dash='dashdot'
                )
            )
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
    if logscale:
        fig.update_xaxes(type='log')
    if save_path is not None:
        fig.write_image(save_path)
    fig.show()

def calculate_coverage(ranks, num_samples):
    expected_coverage = np.arange(1, num_samples + 1) / (num_samples + 1)
    observed_coverage = np.cumsum(np.histogram(ranks, bins=np.arange(num_samples + 1))[0]) / len(ranks)
    return expected_coverage, observed_coverage

def calculate_confidence_intervals(expected_coverage, num_samples, alpha=0.05):
    lower_bound = binom.ppf(alpha / 2, num_samples, expected_coverage) / num_samples
    upper_bound = binom.ppf(1 - alpha / 2, num_samples, expected_coverage) / num_samples
    return lower_bound, upper_bound

# def plot_coverage(expected_coverage, observed_coverage, lower_bound, upper_bound):
#     plt.figure(figsize=(8, 6))
#     plt.plot(expected_coverage, observed_coverage, label='Observed Coverage')
#     plt.plot(expected_coverage, expected_coverage, '--', label='$y=x$')
#     plt.fill_between(expected_coverage, lower_bound, upper_bound, color='grey', alpha=0.5, label='95\% Confidence Interval')
#     plt.xlabel('Expected Coverage')
#     plt.ylabel('Observed Coverage')
#     plt.title('Expected vs Observed Coverage')
#     plt.legend()
#     plt.grid(True)
#     plt.savefig('simulated_data/power_law/coverage_power_law.png')
#     plt.show()


def plot_sbcc(
        rank_filepath: np.ndarray,
        num_posterior_samples: int = 1000,
        save_path: str = None,
        title='SBCC'
) -> None:
    ranks = np.load(rank_filepath)
    expected_coverage, observed_coverage = calculate_coverage(ranks, num_posterior_samples)
    lower_bound, upper_bound = calculate_confidence_intervals(expected_coverage, len(ranks))
    fig = px.line(
        x=expected_coverage,
        y=observed_coverage,
        title=title,
        labels={'x': 'Expected', 'y': 'Observed'}
    )
    fig.add_shape(
        type='line',
        x0=0,
        y0=0,
        x1=1,
        y1=1,
        line=dict(
            color='black',
            width=2,
            dash='dashdot'
        )
    )
    if save_path is not None:
        fig.write_image(save_path)
    fig.show()


def plot_coverage(
        observed_coverage_filepath,
        expected_coverage: np.ndarray = np.linspace(0, 1, 100),
        save_path: str = None
):
    observed_coverage = np.load(observed_coverage_filepath)
    num_params = observed_coverage.shape[1]
    
    fig = make_subplots(rows=1, cols=num_params, subplot_titles=[f'Parameter {i+1}' for i in range(num_params)])
    
    for i in range(num_params):
        fig.add_trace(
            go.Scatter(
                x=expected_coverage,
                y=observed_coverage[:, i],
                mode='lines',
                name=f'Parameter {i+1}'
            ),
            row=1, col=i+1
        )
        fig.add_shape(
            type='line',
            x0=0,
            y0=0,
            x1=1,
            y1=1,
            line=dict(
                color='black',
                width=2,
                dash='dashdot'
            ),
            row=1, col=i+1
        )
    
    fig.update_layout(
        title_text='Expected vs Observed Coverage for Each Parameter',
        xaxis_title='Expected Coverage',
        yaxis_title='Observed Coverage'
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
        save_path='simulated_data/power_law/plots/c2st_scores_SNPE.png')
    df = load_df('simulated_data/power_law/sequential/meddist_scores_SNPE.csv')
    plot_metrics(
        df=df, 
        metric='meddist', 
        title='Sequential PPC median distance for power law model',
        save_path='simulated_data/power_law/plots/meddist_scores_SNPE.png')
    df = load_df('simulated_data/power_law/v2c2st_scores.csv')
    plot_metrics(
        df=df, 
        metric='c2st', 
        title='C2ST scores for power law model, lower dim',
        save_path='simulated_data/power_law/plots/v2c2st_scores.png',
        logscale=True,
        tickvals=[1, 10, 100],
        ticktext=['1k', '10k', '100k'])
    ranks_filepath = 'simulated_data/power_law/ranks_power_law3.npy'
    df = load_df('simulated_data/power_law/v2meddist_scores.csv')
    plot_metrics(
        df=df,
        metric='meddist',
        title='PPC median distance for power law model, lower dim',
        save_path='simulated_data/power_law/plots/v2meddist_scores.png',
        logscale=True,
        tickvals=[1, 10, 100],
        ticktext=['1k', '10k', '100k'])
    

    plot_sbcc(
        rank_filepath=ranks_filepath,
        num_posterior_samples=1000,
        save_path='simulated_data/power_law/plots/coverage_power_law_1k_sims.png',
        title='SBCC for power law model, 1k sims'
    )
    ranks_filepath = 'simulated_data/power_law/ranks_power_law2.npy'
    plot_sbcc(
        rank_filepath=ranks_filepath,
        num_posterior_samples=1000,
        save_path='simulated_data/power_law/plots/coverage_power_law_5k_sims.png',
        title='SBCC for power law model, 5k sims'
    )
    ranks_filepath = 'simulated_data/power_law/ranks_power_law.npy'
    plot_sbcc(
        rank_filepath=ranks_filepath,
        num_posterior_samples=1000,
        save_path='simulated_data/power_law/plots/coverage_power_law_10k_sims.png',
        title='SBCC for power law model, 10k sims'
    )

    observed_coverage_filepath = 'simulated_data/power_law/coverage_vals.npy'
    plot_coverage(
        observed_coverage_filepath,
        save_path='simulated_data/power_law/plots/coverage_power_law.png'
    )