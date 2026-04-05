import pickle
import socket
import pandas as pd
import requests
import base64

def sql2df(sql: str, region: str) -> pd.DataFrame:
    """
    在指定国家数仓执行SQL语句，并返回DataFrame
    
    :param sql: 想要执行的SQL语句，建议使用三引号包裹
    :param region: 🇲🇽墨西哥:mx 🇻🇳越南:vn 🇹🇭泰国:th 🇨🇴哥伦比亚:co
    :return: DataFrame
    
    :Example:
    
    >>> sql = '''
    >>>     SELECT
    >>>         *
    >>>     FROM
    >>>         dwd_order_borrow_mx_full
    >>>     WHERE dt = '2021-01-01'
    >>>     LIMIT 100
    >>> '''
    >>> df = sql2df(sql, 'mx')
    """

    # Flask 应用的 URL
    url = 'http://common-api.xm-risk.com/odps'

    # 要执行的 SQL 语句和国家缩写
    data = {
        # 要执行的sql语句
        'sql': sql,
        # 选择国家 墨西哥:mx 越南:vn
        'region': region,
        'host_name': socket.gethostname()
    }

    # 发起请求
    response = requests.post(url, json=data)
    df = None

    # 检查响应状态码
    if response.status_code == 200:
        try:
            # 如何正常执行，将响应数据转化为DataFrame
            df = pickle.loads(response.content)
        except:
            print(response.content)
    else:
        # 如果执行失败打印失败原因
        print('Error:', response.text)

    return df.reset_index(drop=True)

def upload_to_odps(df: pd.DataFrame, table_name: str, region: str, mode: str = 'overwrite') -> None:
    """
    将 DataFrame 上传到 ODPS，df 中列的顺序必须和目标表的字段顺序相同

    :param df: 待上传的DataFrame
    :param table_name: 目标表名
    :param region: 🇲🇽墨西哥:mx 🇻🇳越南:vn 🇹🇭泰国:th 🇨🇴哥伦比亚:co
    :param mode: 覆盖原表：overwrite 在原表基础上追加：insert
    :return: None
    
    :Example:
    >>> import pandas as pd
    >>> df = pd.DataFrame({'id': [1, 2, 3], 'name': ['a', 'b', 'c']})
    >>> upload_to_odps(df, 'tmp_upload_test', 'mx')
    """

    # Flask 应用的 URL
    url = 'http://common-api.xm-risk.com/odps/upload'
    df = df.reset_index(drop=True)

    # 要执行的 SQL 语句和国家缩写 mx vn th
    data = {
        "data": base64.b64encode(pickle.dumps(df)).decode('utf-8'),
        "table_name": table_name,
        "region": region,
        'host_name': socket.gethostname(),
        'mode': mode
    }

    response = requests.post(url, json=data)
    print(response.text)
    res = response.json()

    if response.status_code == 200:
        print(f'Upload to {table_name} success!')
    else:
        print('Error:', res.get('error'))
        return

def upload_to_odps_partition(df: pd.DataFrame, table_name: str, region: str, dt: str, mode='insert'):
    """
    将 DataFrame 上传到 ODPS，df 中列的顺序必须和目标表的字段顺序相同

    :param df: 待上传的DataFrame
    :param table_name: 目标表名
    :param region: 🇲🇽墨西哥:mx 🇻🇳越南:vn 🇹🇭泰国:th 🇨🇴哥伦比亚:co
    :param dt: 分区日期 yyyyMMdd
    :param mode: 覆盖原分区：overwrite 在原分区上追加：insert
    :return: None

    :Example:
    >>> import pandas as pd
    >>> df = pd.DataFrame({'id': [1, 2, 3], 'name': ['a', 'b', 'c']})
    >>> upload_to_odps_partition(df, 'tmp_upload_test', 'mx','20200101')
    """

    # Flask 应用的 URL
    url = 'http://common-api.xm-risk.com/odps/upload_partition'

    # 要执行的 SQL 语句和国家缩写 mx vn th
    data = {
        "data": base64.b64encode(pickle.dumps(df)).decode('utf-8'),
        "table_name": table_name,
        "region": region,
        'host_name': socket.gethostname(),
        'mode': mode,
        'dt': dt
    }

    response = requests.post(url, json=data)
    res = response.json()

    if response.status_code == 200:
        print(f'Upload to {table_name} success!')
    else:
        print('Error:', res.get('error'))