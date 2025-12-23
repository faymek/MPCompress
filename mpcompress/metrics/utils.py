import torch
import pandas as pd


class DictAverageMeter:
    """Averaging meter for dictionary records.

    This class accumulates numeric values from dictionary records and computes
    their averages. It filters out non-numeric values and handles various numeric
    types including scalars, single-element tensors, and objects with arithmetic
    operations.

    Example:

        Input record: {"key1": value1, "key2": value2, ...}
        Average record: {"key1": avg_value1, "key2": avg_value2, ...}
    """

    def __init__(self):
        """Initialize DictAverageMeter.

        Creates an empty meter with no accumulated values.
        """
        self.meter = None
        self.count = 0

    def _only_keep_number_items(self, record):
        """Filter record to keep only numeric items.

        Args:
            record (dict): Input dictionary with potentially mixed types.

        Returns:
            dict: Dictionary containing only numeric values. Supports:
                - int and float
                - Single-element torch.Tensor (converted to Python scalar)
                - Objects with __add__ and __truediv__ methods
        """
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
        """Update the meter with a new record.

        Args:
            record (dict): Dictionary of key-value pairs to accumulate.
                Non-numeric values are filtered out.
            n (int, optional): Number of samples this record represents.
                Used for weighted averaging. Defaults to 1.
        """
        record = self._only_keep_number_items(record)
        if self.meter is None:
            self.meter = record
        else:
            for key, value in record.items():
                self.meter[key] += value
        self.count += n

    def average(self):
        """Compute the average of all accumulated values.

        Returns:
            average (dict): Dictionary mapping keys to their average values.
                Each value is computed as: accumulated_value / count

        Raises:
            AttributeError: If no records have been updated yet (meter is None).
        """
        record_avg = {}
        for key in self.meter.keys():
            record_avg[key] = self.meter[key] / self.count
        return record_avg


class DataFrameRecords:
    """Record storage and averaging using pandas DataFrame.

    This class stores records in a pandas DataFrame with _id as the index,
    allowing records to be updated by their _id. It supports computing averages
    across all stored records.

    Example:

        Input record: {"_id": id1, "key1": value1, "key2": value2, ...}
        Average record: {"key1": avg_value1, "key2": avg_value2, ...}

    The record can be updated by _id. Implemented using pandas DataFrame,
    supporting row updates by _id.
    """

    def __init__(self):
        """Initialize DataFrameRecords.

        Creates an empty DataFrame for storing records. The _id field will
        be used as the DataFrame index.
        """
        self.df = pd.DataFrame()  # Use DataFrame to store data, with _id as index

    def __len__(self):
        """Get the number of records.

        Returns:
            length (int): Number of records stored in the DataFrame.
        """
        return len(self.df)

    def get_record_by_id(self, _id):
        """Retrieve a record by its _id.

        Args:
            _id (str): The identifier of the record to retrieve.

        Returns:
            record (dict): Dictionary representation of the record with the given _id.

        Raises:
            KeyError: If the _id does not exist in the DataFrame.
        """
        return self.df.loc[_id].to_dict()

    def update(self, record):
        """Update or insert a record by _id.

        If a record with the given _id exists, it will be updated with new values.
        Otherwise, a new record will be inserted. The _id field is used as the
        DataFrame index and is removed from the record data.

        Args:
            record (dict): Dictionary containing "_id" and other key-value pairs.
                The "_id" field is required and will be used as the DataFrame index.

        Raises:
            AssertionError: If the record does not contain "_id" key.
            KeyError: If the _id does not exist in the DataFrame.
        """
        assert "_id" in record, "record must contain _id"
        _id = record["_id"]

        # Create new row data (exclude _id, as _id is used as index)
        record.pop("_id")

        if self.df.empty:
            # If DataFrame is empty, create it directly
            self.df = pd.DataFrame([record], index=[_id])
        elif _id in self.df.index:
            # If _id already exists, update the row
            for key, value in record.items():
                self.df.at[_id, key] = value
        else:
            # If _id does not exist, add a new row
            new_df = pd.DataFrame([record], index=[_id])
            self.df = pd.concat([self.df, new_df])

    def average(self):
        """Compute the average of all numeric columns across all records.

        Returns:
            average (dict): Dictionary mapping column names to their mean values.
                Only numeric columns are included in the result.
        """
        return self.df.mean().to_dict()
