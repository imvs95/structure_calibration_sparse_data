import pandas as pd
from celibration import DataTransform, DataTransformFactory

import random
import numpy as np
import datetime
from itertools import chain
import logging


class PluginFactory(DataTransformFactory):
    def create_transform(self, info=dict) -> DataTransform:
        return NoiseSamplerDataTransform(info=info)


class NoiseSamplerDataTransform(DataTransform):
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

        return index

    def assign_noise(
        self,
        percentage,
        dataframe,
        percentage_noise_width=1,
        date_delta=182,
        dict_alternatives=None,
    ):
        """This function assigns noise to an user-defined percentages values from the observed dataframe. Categorial units are replaced
        by an alternative of a selected list. Numerical units are replace by alternative following a Normal distribution.

        Args:
            percentage (int): Percentage of missing values.
            dataframe (dataframe): Dataframe of data to delete values.

        Returns:
            dataframe: Dataframe with noise given the percentage.
        """

        # Convert values of dataframe to one list
        list_data = [
            value for in_list in dataframe.values.tolist() for value in in_list
        ]

        # Determine how many values need to be deleted
        num_choice = int(round(percentage * len(list_data)))

        # Replace value with an alternative
        chosen_indeces = set()
        total_elements = len(list_data)
        for _ in range(num_choice):
            index = self.index_number(total_elements-1, chosen_indeces)
            list_data[index] = self.determine_noise(
                list_data[index], percentage_noise_width, date_delta, dict_alternatives
            )
            chosen_indeces.add(index)

        # Split the list in number of rows of dataframe
        num_rows = len(dataframe)
        elements_per_row = len(list_data) // num_rows
        reshaped_data = np.array(list_data[:num_rows * elements_per_row]).reshape(num_rows, elements_per_row)

        # Reformat list to dataframe
        noise_df = pd.DataFrame(
            data=reshaped_data, columns=dataframe.columns.tolist()
        )

        return noise_df

    def determine_noise(
        self, value, percentage_noise_width, date_delta, dict_alternatives):
        
        """This function determines noise for a specific value. Categorial units are replaced by an alternative
         of a selected list following an Uniform distribution. Numerical units are replaced by alternative
         following a Normal distribution with as mean the value itself and the standard deviation determined by the
         user-defined percentage of noise width. Default is 100% for uniformity.

        Args:
            value (any): Value to add noise to.
            percentage_noise_width (float): Percentage of the value that is marked as the standard deviation.
            date_delta (int): Number of days that is used for the standard deviation of the date.
            dict_alternatives (dict): Dictonairy with column name as key and a list of alternatives as value.

        Returns:
            noise_value                : Value with noise."""

        if type(value) == float or type(value) == int:
            if value >= 0:
                # For the continous numerical values
                noise_value = np.random.normal(value, percentage_noise_width * value)
                return noise_value
            else:
                # For the Latitude and Longitude
                noise_value = -abs(
                    np.random.normal(abs(value), percentage_noise_width * abs(value))
                )
                return noise_value

        if type(value) == datetime.time:
            noise_value_time = self.noise_in_time(value, percentage_noise_width)

            return noise_value_time

        try:
            if (
                type(datetime.datetime.strptime(str(value), "%Y-%m-%d"))
                == datetime.datetime
            ):
                # For Date with the use of proleptic Gregorian ordinal
                date_time = datetime.datetime.strptime(str(value), "%Y-%m-%d")
                date_ordinal = date_time.date().toordinal()
                noise_value = datetime.date.fromordinal(
                    round(np.random.normal(date_ordinal, date_delta))
                ).strftime("%Y-%m-%d")

                return noise_value
        except ValueError:
            pass

        try:
            if (
                type(datetime.datetime.strptime(str(value), "%Y-%m-%d %H:%M:%S"))
                == datetime.datetime
            ):
                # For Date and time with the use of proleptic Gregorian ordinal
                date_time = datetime.datetime.strptime(str(value), "%Y-%m-%d %H:%M:%S")
                date_ordinal = date_time.date().toordinal()
                noise_date = datetime.date.fromordinal(
                    round(np.random.normal(date_ordinal, date_delta))
                )
                noise_time = self.noise_in_time(
                    date_time.time(), percentage_noise_width
                )

                combine_date_time = datetime.datetime.combine(
                    noise_date, noise_time
                ).strftime("%Y-%m-%d %H:%M:%S")

                return combine_date_time
        except ValueError:
            pass

        try:
            if value in chain(*dict_alternatives.values()):
                # For categorial units with alternatives
                key_name_noise = [
                    key
                    for key, list_alt in dict_alternatives.items()
                    if value in list_alt
                ][0]

                # Remove the current value from list of alternatives
                dict_alt_without_value = dict_alternatives[key_name_noise].copy()
                dict_alt_without_value.remove(value)

                noise_value = random.sample(dict_alt_without_value, k=1)[0]

                return noise_value
        except AttributeError:
            pass

        return None

    def noise_in_time(self, value, percentage_noise_width):
        time_in_seconds = int(
            datetime.timedelta(
                hours=value.hour, minutes=value.minute, seconds=value.second
            ).total_seconds()
        )
        noise_value = round(
            np.random.normal(time_in_seconds, percentage_noise_width * time_in_seconds)
        )

        noise_value_hour = noise_value // 3600

        if noise_value_hour < 0:
            days = abs(noise_value_hour // 24)
            noise_value_hour += days * 24

        elif noise_value_hour > 23:
            days = noise_value_hour // 24
            noise_value_hour -= days * 24

        noise_value_time = datetime.time(
            noise_value_hour, (noise_value % 3600) // 60, noise_value % 60
        )

        return noise_value_time

    def transform(self, df_in: pd.DataFrame, debug: bool, **kwargs) -> pd.DataFrame:
        """This functions transforms the dataframe with noise of an user-defined percentage.
        A seed can be set by the user to control the randomness.

        Args:
            df_in (dataframe): Dataset
            percentage (float): Percentage of noise to be implemented

        Returns:
            df_out: Dataset with noise
        """
        if "seed" in kwargs:
            random.seed(kwargs["seed"])
            np.random.seed(kwargs["seed"])

        if "columns_to_transform" in kwargs:
            df_in_transform = df_in[kwargs["columns_to_transform"]]
            df_out_transform = self.assign_noise(kwargs["percentage"], df_in_transform)
            df_out = df_in.copy()
            df_out[kwargs["columns_to_transform"]] = df_out_transform[kwargs["columns_to_transform"]]

        else:
            df_out = self.assign_noise(kwargs["percentage"], df_in)
        # if debug:
        logging.info(
                "Done: Noise sampler percentage {0:.2f} of the values".format(
                    kwargs["percentage"]
                )
            )
        return df_out
