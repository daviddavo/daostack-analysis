from typing import Final

from pathlib import Path

import datetime as dt

import seaborn as sns
import pandas as pd
import numpy as np

IFRAME_TEMPLATE: Final[str] = """
<iframe src="{src}" title="{title}" class="{classes}"></iframe>
"""


def nameColorByNetwork(x, palette=None, namecol="name", other_color=None, latex: bool=False):
    """
    Example: `_table.style.apply(utils.tables.nameColorByNetwork, axis=None)`
    """
    if palette is None:
        palette = sns.color_palette()

    palette = palette.as_hex()
    color_dict = {'mainnet': palette[0], 'xdai': palette[1]}
    if not other_color:
        other_color = palette[2]

    assert namecol in x.columns, 'namecol should be a valid column'

    style = pd.DataFrame('', index=x.index, columns=x.columns)

    if latex:
        # colorstr = 'cellcolor:[HTML]{{{}}}'
        colorstr = 'color:[HTML]{{{}}}'
    else:
        colorstr = 'color:{}'

    style[namecol] = x['network'].apply(lambda c: colorstr.format(color_dict.get(c, other_color)))

    return style


def custom_pct_format(x, prec=2, latex: bool = True):
    if latex:
        pct_wrap = '\\mypc{{{}}}'
    else:
        pct_wrap = '({}%)'

    if x == 1:
        pct = '100'
    else:
        pct = f'{x*100:.2g}'

    return pct_wrap.format(pct)


def add_pct_col(
    df: pd.DataFrame,
    col,
    col_total,
    col_pct=None,
    main_format=lambda x: f'{x:.0f}',
    nan_replace='–',
    latex=False,
):
    col_pct = col_pct or f'{col} (pct)'
    div = df[col]/df[col_total]
    df[col_pct] = df[col].apply(main_format) + ' ' + div.apply(custom_pct_format, latex=latex)
    df.loc[div.isna(), col_pct] = nan_replace
    return df

def save_table(df: pd.DataFrame, relpath: str, title=None, classes="nb2logseq", debug=False):
    path = Path('../logseq')
    assert path.exists(), 'logseq folder should exist'
    
    lspath = Path('.') / 'assets' / 'nb_tables' / relpath
    path = path / lspath
    path.parent.mkdir(parents=True, exist_ok=True)
    title = ""
    
    print(f"saving to {path}")
    
    df.to_html(path)
    iframe_str = IFRAME_TEMPLATE.format(src=lspath, title=title, classes=classes)
    
    print("Use", iframe_str, "to embed it into logseq")
    
    if debug:
        from IPython.display import HTML
        return HTML(IFRAME_TEMPLATE.format(src=path, title=title, classes=classes))
    
    return df

def add_watermark(path: Path, text="Hello World!", fontsize=18):
    from PIL import Image, ImageDraw, ImageFont
    image = Image.open(path)
    font = ImageFont.truetype('LiberationMono-Regular.ttf', size=fontsize)
    
    w, h = image.size
    margin = 5

    
    tl, tt, tr, tb = ImageDraw.ImageDraw(image).textbbox((0,2*margin), text, font)
    tw = tr-tl
    th = tb-tt

    newimg = Image.new(image.mode, (w,h+th+2*margin), (255,255,255))
    newimg.paste(image, (0, th+2*margin))
    
    draw = ImageDraw.Draw(newimg)
    draw.text((margin, margin), text, fill=(0,0,0), font=font) #, anchor='ms')
    
    newimg.save(path)
    
    return newimg

def save_table_image(df: pd.DataFrame, relpath: str, title=None, debug=False, watermark=True):
    import dataframe_image as dfi
    
    path = Path('../logseq')
    assert path.exists(), 'logseq folder should exist'
    
    lspath = Path('.') / 'assets' / 'nb_tables' / relpath
    path = path / lspath
    path.parent.mkdir(parents=True, exist_ok=True)
    
    print(f"saving to {path}")
    
    dfi.export(df, path)
    if watermark == True:
        add_watermark(path, f"{path}\nGenerated on: {dt.date.today().isoformat()}\nTitle: {title}")
    elif watermark:
        add_watermark(path, watermark)
    
    print(f"Use ![{title}]({lspath}) to embed it in logseq")
    
    if debug:
        from IPython.display import Image
        return Image(path)
    
    return df
    