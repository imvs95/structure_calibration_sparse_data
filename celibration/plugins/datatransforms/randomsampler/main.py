import pandas as pd
from celibration import DataTransform, DataTransformFactory


class PluginFactory(DataTransformFactory):
    def create_transform(self, info=dict) -> DataTransform:
        return RandomSamplerDataTransform(info=info)


class RandomSamplerDataTransform(DataTransform):
    def transform(self, df_in: pd.DataFrame, debug: bool, **kwargs) -> pd.DataFrame:
        df_out = df_in.sample(n=kwargs["n"], random_state=kwargs["random_state"])

        return df_out
