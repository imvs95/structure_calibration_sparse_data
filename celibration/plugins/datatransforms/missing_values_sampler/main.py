import pandas as pd
from celibration import DataTransform, DataTransformFactory

import random
import numpy as np
import logging


class PluginFactory(DataTransformFactory):
    def create_transform(self, info=dict) -> DataTransform:
        return MissingValuesSamplerDataTransform(info=info)


class MissingValuesSamplerDataTransform(DataTransform):
    def index_number(self, max_number, list_index_chosen):
        """This function randomly defines an index number and ensures that no index number is assigned twice.

        Args:
            max_number (int): Higher bound of the index number.
            list_index_chosen (list/set): List of indeces that have already been chosen.

        Returns:
            int: Index number
        """ 
        index = random.randint(0, max_number)

        if index in list_index_chosen:
            return self.index_number(max_number, list_index_chosen)

        # list_index_chosen.append(index)
        return index

    def delete_values_completely_random(self, dataframe, percentage):
        """This function deletes an user-defined percentages values from the observed dataframe, based on the Missing
        Value Completely Random type. It replaces the values with a None value.

        Args:
            percentage (float): Percentage of missing values.
            dataframe (dataframe): Dataframe of data to delete values.

        Returns:
            dataframe: Dataframe with missing values given the percentage.
        """

        # Convert values of dataframe to one list
        list_data = [
            value for in_list in dataframe.values.tolist() for value in in_list
        ]

        # Determine how many values need to be deleted
        num_choice = int(round(percentage * len(list_data)))

        # Replace value of list with NaN value
        chosen_indeces = set()
        total_elements = len(list_data)
        for _ in range(num_choice):
            index = self.index_number(total_elements-1, chosen_indeces)
            list_data[index] = None
            chosen_indeces.add(index)


        # Split the list in number of rows of dataframe
        num_rows = len(dataframe)
        elements_per_row = len(list_data) // num_rows
        reshaped_data = np.array(list_data[:num_rows * elements_per_row]).reshape(num_rows, elements_per_row)

        # Reformat list to dataframe
        missing_df = pd.DataFrame(
            data=reshaped_data, columns=dataframe.columns.tolist()
        )

        return missing_df

    def transform(self, df_in: pd.DataFrame, debug: bool, **kwargs) -> pd.DataFrame:
        """This functions transforms the dataframe with missing values of an user-defined percentage.
        A seed can be set by the user to control the randomness.

        Args:
            df_in (dataframe): Dataset
            percentage (float): Percentage of missing values to be implemented

        Returns:
            df_out: Dataset with missing values
        """
        if "seed" in kwargs:
            random.seed(kwargs["seed"])

        if "columns_to_transform" in kwargs:
            df_in_transform = df_in[kwargs["columns_to_transform"]]
            df_out_transform = self.delete_values_completely_random(df_in_transform, kwargs["percentage"])
            df_out = df_in.copy()
            df_out[kwargs["columns_to_transform"]] = df_out_transform[kwargs["columns_to_transform"]]

        else:
            df_out = self.delete_values_completely_random(df_in, kwargs["percentage"])
        # if debug:
        logging.info(
                "Done: Missing values sampler with percentage {0:.2f}".format(
                    kwargs["percentage"]
                )
            )
        return df_out
