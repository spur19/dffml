"""
Expose Pandas DataFrame as DFFML Source
"""
from collections import OrderedDict
from typing import Dict, List, AsyncIterator


from ..record import Record
from ..base import config, field
from ..util.entrypoint import entrypoint
from .source import BaseSourceContext, BaseSource


class DataFrameSourceContext(BaseSourceContext):
    async def update(self, record: Record):

        df = self.parent.config.dataframe

        # Store feature data
        feature_columns = self.parent.config.feature_cols
        feature_data = OrderedDict.fromkeys(feature_columns)
        feature_data.update(record.features(feature_columns))

        for col in feature_columns:
            df.loc[record.key, col] = feature_data[col]

        # Store prediction
        try:
            prediction = record.prediction("target_name")
            prediction_columns = self.parent.config.prediction_cols
            prediction_data = OrderedDict.fromkeys(prediction_columns)
            prediction_data.update(prediction.dict())

            for col in prediction_columns:
                df.loc[record.key, col] = prediction_data[col]

        except KeyError:
            pass

    async def records(self) -> AsyncIterator[Record]:
        for row in self.parent.config.dataframe.itertuples():
            features = row._asdict()
            del features["Index"]
            yield Record(str(row.Index), data={"features": features})

    async def record(self, key: str) -> Record:
        return Record(
            str(key),
            data={"features": {**self.parent.config.dataframe.iloc[int(key)]}},
        )


@config
class DataFrameSourceConfig:
    dataframe: "pandas.DataFrame" = field("The pandas DataFrame to proxy")
    feature_cols: List[str] = field(
        "Feature columns whose values we have to update"
    )
    prediction_cols: List[str] = field(
        "Prediction columns whose values we have to update"
    )


@entrypoint("dataframe")
class DataFrameSource(BaseSource):
    """
    Proxy for a pandas DataFrame
    """

    CONFIG = DataFrameSourceConfig
    CONTEXT = DataFrameSourceContext
