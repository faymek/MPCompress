import torch
import pandas as pd


class DictAverageMeter:
    """计算平均值
    input record: {"key1": value1, "key2": value2, ...}
    average record: {"key1": value1, "key2": value2, ...}
    """

    def __init__(self):
        self.meter = None
        self.count = 0

    def _only_keep_number_items(self, record):
        new_record = {}
        for key, value in record.items():
            if isinstance(value, (int, float)):
                new_record[key] = value
            elif isinstance(value, torch.Tensor) and value.numel() == 1:
                new_record[key] = value.item()
            elif hasattr(value, "__add__") and hasattr(value, "__truediv__"):
                new_record[key] = value
        return new_record

    def update(self, record, n=1):
        record = self._only_keep_number_items(record)
        if self.meter is None:
            self.meter = record
        else:
            for key, value in record.items():
                self.meter[key] += value
        self.count += n

    def average(self):
        record_avg = {}
        for key in self.meter.keys():
            record_avg[key] = self.meter[key] / self.count
        return record_avg


class DataFrameRecords:
    """计算平均值
    input record: {"key1": value1, "key2": value2, ...}
    average record: {"key1": value1, "key2": value2, ...}
    the record can be updated by _id
    基于 pandas DataFrame 实现，支持按 _id 更新行记录
    """

    def __init__(self):
        self.df = pd.DataFrame()  # 使用 DataFrame 存储数据，_id 作为索引

    def __len__(self):
        return len(self.df)

    def get_record_by_id(self, _id):
        return self.df.loc[_id].to_dict()

    def update(self, record):
        """根据 _id 更新或插入记录，将可累计类型转为数值"""
        assert "_id" in record, "record must contain _id"
        _id = record["_id"]

        # 创建新行数据（排除 _id，因为 _id 作为索引）
        record.pop("_id")

        if self.df.empty:
            # 如果 DataFrame 为空，直接创建
            self.df = pd.DataFrame([record], index=[_id])
        elif _id in self.df.index:
            # 如果 _id 已存在，更新该行
            for key, value in record.items():
                self.df.at[_id, key] = value
        else:
            # 如果 _id 不存在，添加新行
            new_df = pd.DataFrame([record], index=[_id])
            self.df = pd.concat([self.df, new_df])

    def average(self):
        return self.df.mean().to_dict()