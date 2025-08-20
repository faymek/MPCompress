import torch


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
