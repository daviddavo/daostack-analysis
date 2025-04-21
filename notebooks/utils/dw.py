import datetime as dt

from dao_analyzer.web.apps.daostack.data_access.daos.metric import srcs as DAOSTACK

def get_date() -> dt.datetime:
    with open(DAOSTACK.DATAWAREHOUSE / 'update_date.txt', 'r') as f:
        return dt.datetime.fromisoformat(f.readline())
