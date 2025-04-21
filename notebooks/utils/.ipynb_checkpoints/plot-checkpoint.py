import datetime as dt
from typing import Any, List
from pathlib import Path
import copy

import pandas as pd

import plotly.graph_objects as go
import plotly.io as pio
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn.metrics

import utils
from utils import dw

LAST_DATA_DATE: dt.date =dw.get_date()
default_width = 1200
default_height = 300
default_heatmap_cmap = 'mako_r'

def classification_report_2x2(df: pd.DataFrame, suptitle: str='Classification report', supy: str='network'):
    Y_AVAILABLE = sorted(df[supy].unique())
    PREDICTORES = ['stakes', 'boosted']
    
    fig, axs = plt.subplots(ncols=len(Y_AVAILABLE), nrows=2, sharex=True, sharey=True, tight_layout=True)
    
    fig.suptitle(suptitle)
    fig.supxlabel("Predictor")
    fig.supylabel(supy.capitalize())
    
    print("y_true = approved, y_pred = boosted")
    print(sklearn.metrics.classification_report(y_true=df['approved'], y_pred=df['boosted'], zero_division=0))
    for i, n in enumerate(Y_AVAILABLE):
        dfprn = df[df[supy] == n]
        print(f"## {n} only ##")
        print(sklearn.metrics.classification_report(y_true=dfprn['approved'], y_pred=dfprn['boosted'], zero_division=0))
        for j, p in enumerate(PREDICTORES):
            ct = pd.crosstab(dfprn['approved'], dfprn[p])
            if True not in ct.columns:
                ct[True] = 0
                
            sns.heatmap(ct, annot=True, fmt='d', ax=axs[i][j])

            # Hide labels on the middle
            axs[i][j].set_xlabel(axs[i][j].get_xlabel() if i != 0 else '')
            axs[i][j].set_ylabel(f"{n}\n{axs[i][j].get_ylabel()}" if j == 0 else '')
            
    plt.close()
    return fig

def save_fig(fig: go.Figure | Any, relpath: str | Path, show_title: bool = False, showlegend: bool = None, debug: bool = False, **kwargs):
    """ Saves the figure `fig` to the path `<logseq_dir>/relpath`
    """
    path = Path('../logseq')
    assert path.exists(), 'logseq folder should exist'
    
    lspath = Path('.') / 'assets' / 'nb_figures' / relpath
    path = path / lspath
    path.parent.mkdir(parents=True, exist_ok=True)
    title = ""
    
    print(f"saving to {path}")
    
    # PLOTLY
    if isinstance(fig, go.Figure):
        kwargs.setdefault('width', default_width)
        kwargs.setdefault('height', default_height)
        
        auxfig = go.Figure(fig)
        title = fig.layout.title.text
        
        if not show_title and title:
            auxfig.update_layout(
                title=None, 
                showlegend=showlegend, 
                margin=go.layout.Margin(t=20, pad=0),
                annotations=[{
                    'name': 'title',
                    'text': title,
                    'opacity': 0.1,
                    'textangle': -10,
                    'xref': 'paper',
                    'yref': 'paper',
                    'font': {'color': 'red', 'size': 20, 'family': 'monospace'},
                    'xanchor': 'center',
                    'x': 0.5,
                    'y': 0.35,
                    'showarrow': False,
            }])
        
        auxfig.write_image(path, **kwargs)
    # MATPLOTLIB | SEABORN
    elif isinstance(fig, plt.Axes | sns.FacetGrid | plt.Figure):
        kwargs.setdefault('bbox_inches', 'tight')
        
        if isinstance(fig, plt.Axes):
            print('Figure is Axes')
            title = fig.get_title()

            if not show_title:
                fig.set_title("")
            closefig = fig.get_figure()
            auxfig = copy.deepcopy(fig.get_figure())
            
            if not show_title:
                auxfig.suptitle("")
        elif isinstance(fig, plt.Figure):
            print('Figure is Figure')
            title = fig._suptitle.get_text()
            closefig = fig
            auxfig = fig

            if not show_title:
                auxfig.suptitle("")
        else:
            print('Figure is other')
            title = fig.axes[0][0].get_title() or 'FacetGrid'
            auxfig = copy.deepcopy(fig.figure)
            closefig = fig.figure # For the `if debug` part

            if not show_title:
                auxfig.suptitle("")
            
        if showlegend is not None:
            for ax in auxfig.axes:
                    ax.get_legend().set_visible(showlegend)
                    
        if title:
            auxfig.text(0.5, 0.35, title, fontsize=10, color='red', alpha=0.5, ha='center', va='center', rotation=30)
        
        auxfig.text(0.5, 0.5, f'Borrador {dt.date.today()}', fontsize=32, color='gray', alpha=0.5, ha='center', va='center', rotation=30)
        auxfig.savefig(path, **kwargs)
        plt.close(auxfig)
        
        if debug:
            plt.close(closefig)
    else:
        raise NotImplementedError(f'save_fig not implemented for type {type(fig)}')
        
    print(f"Use ![{title}](../{lspath}) to embed it in logseq")
    
    if debug:
        from IPython.display import Image
        return Image(path)
    
    return fig

def get_network_color(network: str, palette=None, format=None):
    if palette is None:
        palette = sns.color_palette()
    
    networks_idx = {
        'mainnet': palette[0], 
        'xdai': palette[1],
    }
    other_color = palette[2]
    
    color = networks_idx.get(network, other_color)

    if format == 'rgb' or format == 'plotly':
        return 'rgb('+ ",".join(f'{c*255:.0f}' for c in color) +')'
    else:
        return color

def _get_label_from_row(row: pd.Series, palette = sns.color_palette(), namecol='name') -> str:
    assert namecol in row, 'namecol should be a valid column'
    return plt.Text(text=row[namecol], color=get_network_color(row['network']))

def get_colored_dao_labels_from_ids(df: pd.DataFrame, keep_unregistered_names: bool = False) -> List[plt.Text]:
    # Use ['network', 'dao'] to get the DAO name and use a orange/blue color depending on mainnet/ethereum
    return list(utils.append_dao_names(df[['network', 'dao']], keep_unregistered_names = keep_unregistered_names).apply(_get_label_from_row, axis='columns'))

def get_colored_dao_labels_from_network(df: pd.DataFrame, namecol='name') -> List[plt.Text]:
    return list(df.apply(_get_label_from_row, axis='columns', namecol=namecol))

BORRADOR_ANNOTATION = go.layout.Annotation(
    name = 'date',
    text = f'Borrador {dt.date.today()}',
    textangle = -10,
    opacity = 0.1,
    font = go.layout.annotation.Font(size=37, color='black'),
    xref = 'paper',
    yref = 'paper',
    x = 0.5,
    y = 0.5,
    showarrow = False,
)

pio.templates['custom'] = go.layout.Template(
    layout = go.Layout(
        template='seaborn',
        margin=go.layout.Margin(b=50,l=50,r=50,t=50),
        annotations = [ BORRADOR_ANNOTATION ],
    )
)

pio.templates['custom_timeseries_wide'] = go.layout.Template(
    layout = go.Layout(
        template = 'seaborn',
        margin=go.layout.Margin(b=50, l=50, r=50, t=50),
        showlegend=False,
        xaxis={
            'tickmode': 'array',
            'tickvals': pd.date_range(start=dt.date(2019, 4, 1), end=LAST_DATA_DATE, freq='3M'),
            'tickformat': '%b \'%y'
        },
        annotations = [ BORRADOR_ANNOTATION ],
    ),
)
