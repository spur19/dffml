
import pathlib
import pandas as pd
import numpy as np
from typing import AsyncIterator, Type

from sktime.datasets import load_airline
from sktime.forecasting.base import ForecastingHorizon
from sktime.forecasting.exp_smoothing import ExponentialSmoothing
from sktime.performance_metrics.forecasting import sMAPE, smape_loss



from dffml import (
    Accuracy,
    Feature,
    Features,
    ModelNotTrained,
    Record,
    SimpleModel,
    SourcesContext,
    config,
    entrypoint,
    field,
)

@config
class ExpModelConfig:
    features: Features = field("Features to train on")
    predict: Feature = field("Label or the value to be predicted")
    directory: pathlib.Path = field("Directory where state should be saved")
    spd: int = field("Seasonal Periodicity", default=12)
    fh: int = field("Forecasting horizon",default=spd)


@entrypoint("expsmoothing")
class ExpModel(SimpleModel):

    # The configuration class needs to be set as the CONFIG property
    CONFIG: Type = ExpModelConfig

    async def train(self, sources: SourcesContext) -> None:
        
        # X and Y data
        X = []
        # Go through all records that have the feature we're training on and the
        # feature we want to predict.
        async for record in sources.with_features(
            self.features #+ 
             #[self.parent.config.predict.name]
        ):
            record_data = []
            for feature in record.features(self.features).values():
                record_data.extend(
                    [feature] if np.isscalar(feature) else feature
                )

            X.append(record_data)
           

        
        X = pd.DataFrame(X)

        spd = self.parent.config.spd

        # Use self.logger to report how many records are being used for training
        self.logger.debug("Number of training records: %d", len(X))
        
        self.forecaster = ExponentialSmoothing(trend="add", seasonal="multiplicative",sp=12)
        self.forecaster.fit(X)

        # Save the trained model
        joblib.dump(self.forecaster, str(self.forecaster_filepath))

    async def accuracy(self, sources: SourcesContext) -> Accuracy:
        """
        Evaluates the accuracy of the model by gathering predictions of the test data
        and comparing them to the provided results.
        
        We will use the sMAPE (symmetric mean absolute percentage error) to quantify the accuracy of our forecasts. 
        A lower sMAPE means higher accuracy.
        """
        if not self.forecaster:
            raise ModelNotTrained("Train the model before assessing accuracy")

        # Get data
        #input_data = await self.get_input_data(sources)

        X_test = []
        Y_test = []
        # Make predictions
        async for record in sources.with_features(
            self.features + [self.parent.config.predict.name]
        ):
            record_data = []
            for feature in record.features(self.features).values():
                record_data.extend(
                    [feature] if np.isscalar(feature) else feature
                )

            X_test.append(record_data)
            Y_test.append(record.feature(self.parent.config.predict.name))

        
        X_test = pd.DataFrame(X_test)

        foh = ForecastingHorizon(X_test.index, is_relative=False)

        y_pred = self.forecaster.predict(foh)

        return smape_loss(X, y_pred)

    async def predict(self, sources: SourcesContext) -> AsyncIterator[Record]:

        """
        Uses saved model to make predictions for a forecast horizon
        """
        if not self.forecaster:
            raise ModelNotTrained(
                "Train the model first before getting predictions"
            )
        
        fh = self.parent.config.fh

        predictions = self.forecaster.predict(fh)

        return predictions    

