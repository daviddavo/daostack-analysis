from typing import Any

from pathlib import Path
import datetime as dt
import warnings

import pandas as pd
from pandas.api.extensions import register_series_accessor

from plotly import graph_objects as go

from dao_analyzer.web.apps.daostack.data_access.daos.metric import srcs as DAOSTACK

from . import plot
from . import tables
from . import functions
from . import dw

PICKLE_PATH = Path('.pickles')
DEFAULT_REGISTERED_ONLY = True
""" Date when we consider that the platform ceased to be
used for its original purpose """
DEGRADATION_DATE = dt.datetime(2021,6,1)
# Should be compared with dfp['executedAt']

def change_unregistered_names(df: pd.DataFrame) -> pd.DataFrame:
    """ Changes the name of DAOs with unregistered """
    
    unRegmsk = df['register'] == 'unRegistered'
    dfUnReg = df[unRegmsk]
    
    vc = dfUnReg.name.value_counts()
    df['originalName'] = df['name']
    def _aux_change_name(row: pd.Series) -> pd.Series:
        # TODO: If there is more than one unRegistered DAO with the same name, use the year or any other id
        if row['register'] == 'unRegistered':
            if vc[row['name']] > 1:
                row['name'] = f'{row["name"]} (unregistered {row["dao"][:6]})'
            else:
                row['name'] = row['name'] + ' (unregistered)'

        return row
    
    return df.apply(_aux_change_name, axis='columns')

def append_dao_names(df: pd.DataFrame, keep_unregistered_names: bool = False) -> pd.DataFrame:
    dfn = get_df('dfd').reset_index()[['network', 'dao', 'name', 'register', 'group']]
    
    if not keep_unregistered_names:
        dfn = change_unregistered_names(dfn)
    
    idx = df.index
    return df.merge(dfn.drop(columns=['register']), on=['network', 'dao'], how='left').set_index(idx)

def append_dao_info(df: pd.DataFrame, keep_unregistered_names: bool = True) -> pd.DataFrame:
    DAO_INFO_COLS = ['group', 'name', 'register']
    dfd = get_df('dfd')[DAO_INFO_COLS].reset_index() # pd.read_feather(DAOSTACK.DAOS, columns=['network', 'dao'] + DAO_INFO_COLS)
    if not keep_unregistered_names:
        dfd = change_unregistered_names(dfd)
    return df.merge(dfd, on=['network', 'dao'], how='left', suffixes=(None, "_di"))

def getMonthsSince(series, since='2023-03-01'):
    """ Count months between first activity (assumed creation date) and today """
    
    # Getting difference in DateOffset MonthEnds
    monthEnds = pd.to_datetime(since).to_period('M') - series.dt.to_period('M')
    # Convert to integer
    return monthEnds.dropna().apply(lambda x: x.n)

def colAutoDatetime(s: pd.Series):
    s = s.astype('Int64')
    maxdate = s.max()
    
    if maxdate < pd.Timestamp.max.timestamp():
        return pd.to_datetime(s, unit='s', origin='unix')
    elif not s.isna().any():
        return s.apply(dt.datetime.fromtimestamp)
    else:
        raise ValueError("Series contains nan values when trying to convert to datetime")

def dfAtToDatetime(df: pd.DataFrame) -> pd.DataFrame:
    """ Convert all fields ending in At (like createdAt) to datetime """
    columns = [ x for x in df.columns if x.endswith('At') ]
    
    df[columns] = df[columns].apply(colAutoDatetime)
    
    return df

def get_df(df_name: str, registered_only: bool = None, **kwargs) -> pd.DataFrame:
    columns = kwargs.pop('columns', None)
    df = pd.read_pickle(PICKLE_PATH / (df_name + '.pickle'), **kwargs)
    if columns:
        df = df[columns]
        
    if registered_only is None:
        if df_name.startswith('dfg'):
            registered_only = False
        else:
            registered_only = DEFAULT_REGISTERED_ONLY
    
    if registered_only:
        allowed_values = ['registered', 'unRegistered']
        
        registerCol = None
        if 'daoRegister' in df.columns:
            registerCol = 'daoRegister'
        elif 'register' in df.columns:
            registerCol = 'register'
            
        if registerCol:
            mask = df[registerCol].isin(allowed_values)
            # print(f"Using the {registerCol} method")
        else:
            dfd = pd.read_feather(DAOSTACK.DAOS, columns=['network', 'dao', 'register'])
            dfd = dfd[dfd['register'].isin(allowed_values)]
            
            assert not dfd.empty, "DAOs dataframe should not be empty"
            
            idx = ['network', 'dao']
            added_cols = []
            _aux = df
            for i in idx:
                if i not in _aux.columns:
                    _aux[i] = df.index.get_level_values(i)
                    added_cols.append(i)

            mask = _aux[idx].apply(tuple,1).isin(dfd[idx].apply(tuple,1))
            
        df = df[mask]
            
        if registerCol:
            assert (df[registerCol].isin(allowed_values)).all()
        
    return df

get_dw_date = dw.get_date

def dropDAOs(df: pd.DataFrame, nprop = 2, nusers = 3, nvoters = 3) -> pd.DataFrame:
    """ Drop DAOs according to the paper policy """
    dfd = get_df('dfd')
    allowed_msk = (dfd['nproposals'] >= nprop) & (dfd['hnusers'] >= nusers) & (dfd['nvoters'] >= nvoters)
    allowed = dfd[allowed_msk]
    
    prevlen = len(df.index)
    df = df[df.index.isin(allowed.index)].copy()
    
    if prevlen == len(df.index):
        warnings.warn('No DAOs dropped in dropDAOs')
    if len(df.index) != allowed_msk.sum():
        warnings.warn('Some DAOs were already filtered in dropDAOs')
        
    return df

def dropDAOsTable(df: pd.DataFrame, n=20) -> pd.DataFrame:
    """ Drop DAOs to make it displayable in a table. See table "Resumen de DAOs" """
    dfd = get_df('dfd')
    allowed = dfd.nlargest(n, 'nproposals')
    
    prevlen = len(df.index)
    df = dropDAOs(df)
    df = df[df.index.isin(dfd.index)]
    
    if prevlen == len(df.index):
        warnings.warn('No DAOs dropped in dropDAOsTable')
    if len(df.index) != 20:
        warnings.warn('Some DAOs were already filtered in dropDAOsTable')
    
    return df

def get_nunique_in_dao(s: pd.Series) -> int:
    """ Obtains the number of nunique addresses that voted/proposed, which are also in the dao 
    **Important**: Must be used with .apply, not with .agg because it relies in the series name
    """
    dfm = pd.read_feather(DAOSTACK.REP_MINTS).set_index(['network', 'dao']).sort_index()
    if s.empty or not s.name in dfm.index: return 0

    daoholders = dfm.loc[s.name]['address']
    
    return len(set(s).intersection(daoholders))

def get_nunique_in_group(s: pd.Series) -> int:
    """ Obtains the number of nunique addresses that voted/proposed, which are also in the GROUP
    **Important**: Must be used with .apply, not with .agg because it relies in the series name
    """
    dfm = append_dao_info(pd.read_feather(DAOSTACK.REP_MINTS)).set_index('group')
    
    if s.empty or not s.name in dfm.index: return 0

    daoholders = dfm.loc[s.name]['address']
    
    return len(set(s).intersection(daoholders))

@pd.api.extensions.register_series_accessor("ds")
class DAOSeriesAccessor:
    def __init__(self, series):
        self._series = series

    def nunique_in_dao(self):
        return get_nunique_in_dao(self._series)

@pd.api.extensions.register_dataframe_accessor("ds")
class DAODataFrameAccessor:
    def __init__(self, df):
        self._df = df
        
    def append_dao_info(self, *args, **kwargs):
        return append_dao_info(self._df, *args, **kwargs)
    
    def add_pct_col(self, *args, **kwargs):
        return tables.add_pct_col(self._df, *args, **kwargs)
    
save_fig = plot.save_fig
