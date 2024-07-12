"""
Created on: 6-8-2021 10:09

@author: IvS
"""
import itertools
import statistics
import numpy as np
import pandas as pd

from pydsol.pydsol_core.DSOLModel import DSOLModel

from police_simulation_model.model_elements.product import Product
from police_simulation_model.model_elements.supplier import Supplier
from police_simulation_model.model_elements.manufacturer import Manufacturer
from police_simulation_model.model_elements.port import Port
from police_simulation_model.model_elements.wholesales_distributor import WholesalesDistributor
from police_simulation_model.model_elements.retailer import Retailer
from police_simulation_model.model_elements.customer import Customer
from police_simulation_model.model_elements.transfer_location import TransferLocation
from pydsol.pydsol_model.link import Link

from police_simulation_model.model_elements.vehicles.ferry import Ferry
from police_simulation_model.model_elements.vehicles.large_truck import LargeTruck
from police_simulation_model.model_elements.vehicles.small_truck import SmallTruck
from police_simulation_model.model_elements.vehicles.train import Train
from police_simulation_model.model_elements.vehicles.boat import Boat

from police_simulation_model.model_elements.timer import Timer

from pydsol.basic_logger import get_module_logger

logger = get_module_logger(__name__)


class SimModel(DSOLModel):
    def __init__(self, input_parameters, **kwargs):
        super().__init__(input_parameters, **kwargs)
        self.input_parameters = input_parameters

        self.supplier_brazil = None
        self.manufacturer_brazil = None
        self.exporter_brazil = None
        self.importer_rotterdam = None
        self.importer_senegal = None
        self.wholesales_distributor_eindhoven = None
        self.retailer_amsterdam = None
        self.retailer_utrecht = None
        self.retailer_nijmegen = None
        self.customer_local = None
        self.customer_export = None

    def construct_model(self):
        """Model has the time unit km/h"""
        # Reset model
        self.reset_model()

        np.random.seed(self.seed)

        # Set up model
        supplier_brazil = Supplier(self.simulator, location='Brazil',
                                   interarrival_time=self.input_parameters["interarrival_time"])
        manufacturer_brazil = Manufacturer(self.simulator, location='Brazil',
                                           processing_time=3.5,
                                           vehicle_type=SmallTruck,
                                           vehicle_speed=100)

        inland_carrier_1 = Link(self.simulator, supplier_brazil, manufacturer_brazil, 0)

        exporter_brazil = Port(self.simulator, location='Brazil', processing_time=1, vehicle_type=Boat,
                               vehicle_speed=2500)

        inland_carrier_2 = Link(self.simulator, manufacturer_brazil, exporter_brazil, 500)

        importer_rotterdam = Port(self.simulator, location='Rotterdam', processing_time=1, vehicle_type=Train,
                                  vehicle_speed=[50, 100, 100])
        importer_senegal = Port(self.simulator, location='Senegal', processing_time=1, vehicle_type=LargeTruck,
                                vehicle_speed=70)

        # Uses input parameters
        transporter = Link(self.simulator, exporter_brazil, importer_rotterdam, 3256,
                           selection_weight=self.input_parameters["route_via_rotterdam"])
        transporter = Link(self.simulator, exporter_brazil, importer_senegal, 9779,
                           selection_weight=1 - self.input_parameters["route_via_rotterdam"])

        wholesales_distributor_eindhoven = WholesalesDistributor(self.simulator,
                                                                 minimum_processing_time=(50 / 60),
                                                                 mode_processing_time=(70 / 60),
                                                                 maximum_processing_time=(100 / 60),
                                                                 location='Eindhoven', vehicle_type=SmallTruck,
                                                                 divide_quantity=3)

        inland_carrier_3 = Link(self.simulator, importer_rotterdam, wholesales_distributor_eindhoven, 135)

        transfer_location_morocco = TransferLocation(self.simulator, location="Morocco", vehicle_type=Ferry)

        inland_carrier_4 = Link(self.simulator, importer_senegal, transfer_location_morocco, 3356)

        transfer_location_spain = TransferLocation(self.simulator, location="Spain")

        ferry = Link(self.simulator, transfer_location_morocco, transfer_location_spain, 43)

        transfer_location_antwerp_belgium = TransferLocation(self.simulator, location="Antwerp")

        inland_carrier_4 = Link(self.simulator, transfer_location_spain, transfer_location_antwerp_belgium, 2164)
        inland_carrier_4 = Link(self.simulator, transfer_location_antwerp_belgium, wholesales_distributor_eindhoven, 88)

        retailer_amsterdam = Retailer(self.simulator, processing_time=0.5, location='Eindhoven')
        retailer_utrecht = Retailer(self.simulator, processing_time=1, location='Utrecht')
        retailer_nijmegen = Retailer(self.simulator, processing_time=2, location='Nijmegen')

        inland_carrier_5 = Link(self.simulator, wholesales_distributor_eindhoven, retailer_amsterdam, 125)
        inland_carrier_6 = Link(self.simulator, wholesales_distributor_eindhoven, retailer_utrecht, 92)
        inland_carrier_7 = Link(self.simulator, wholesales_distributor_eindhoven, retailer_nijmegen, 80)

        customer_local = Customer(self.simulator, type='Local')
        customer_export = Customer(self.simulator, type='Export')

        link_1 = Link(self.simulator, retailer_amsterdam, customer_local, 0, selection_weight=0.05)
        link_2 = Link(self.simulator, retailer_amsterdam, customer_export, 0, selection_weight=0.95)

        link_3 = Link(self.simulator, retailer_utrecht, customer_local, 0, selection_weight=0.05)
        link_4 = Link(self.simulator, retailer_utrecht, customer_export, 0, selection_weight=0.95)

        link_5 = Link(self.simulator, retailer_nijmegen, customer_local, 0, selection_weight=0.05)
        link_6 = Link(self.simulator, retailer_nijmegen, customer_export, 0, selection_weight=0.95)

        # Set up stats
        self.supplier_brazil = supplier_brazil
        self.manufacturer_brazil = manufacturer_brazil
        self.exporter_brazil = exporter_brazil
        self.importer_rotterdam = importer_rotterdam
        self.importer_senegal = importer_senegal
        self.wholesales_distributor_eindhoven = wholesales_distributor_eindhoven
        self.retailer_amsterdam = retailer_amsterdam
        self.retailer_utrecht = retailer_utrecht
        self.retailer_nijmegen = retailer_nijmegen
        self.customer_local = customer_local
        self.customer_export = customer_export

        self.set_output_stats()

        supplier_brazil.create_entities(Product, interarrival_time=self.input_parameters["interarrival_time"])

    @staticmethod
    def reset_model():
        Product.id_iter = itertools.count(1)
        Supplier.id_iter = itertools.count(1)
        Manufacturer.id_iter = itertools.count(1)
        Port.id_iter = itertools.count(1)
        TransferLocation.id_iter = itertools.count(1)
        WholesalesDistributor.id_iter = itertools.count(1)
        Retailer.id_iter = itertools.count(1)
        Customer.id_iter = itertools.count(1)
        Link.id_iter = itertools.count(1)
        Boat.id_iter = itertools.count(1)
        Train.id_iter = itertools.count(1)
        SmallTruck.id_iter = itertools.count(1)
        LargeTruck.id_iter = itertools.count(1)

    def set_output_stats(self):
        timer = Timer(self.simulator, 1)
        timer.set_event(self.supplier_brazil, self.supplier_brazil, "get_hourly_stats")
        timer.set_event(self.manufacturer_brazil, self.manufacturer_brazil, "get_hourly_stats")
        timer.set_event(self.exporter_brazil, self.exporter_brazil, "get_hourly_stats")
        timer.set_event(self.importer_rotterdam, self.importer_rotterdam, "get_hourly_stats")
        timer.set_event(self.importer_senegal, self.importer_senegal, "get_hourly_stats")
        timer.set_event(self.wholesales_distributor_eindhoven, self.wholesales_distributor_eindhoven,
                        "get_hourly_stats")
        timer.set_event(self.retailer_amsterdam, self.retailer_amsterdam, "get_hourly_stats")
        timer.set_event(self.retailer_utrecht, self.retailer_utrecht, "get_hourly_stats")
        timer.set_event(self.retailer_nijmegen, self.retailer_nijmegen, "get_hourly_stats")
        timer.set_event(self.customer_local, self.customer_local, "get_hourly_stats")
        timer.set_event(self.customer_export, self.customer_export, "get_hourly_stats")

    def get_output_statistics(self):
        # Average product time in system
        average_product_time_in_system = statistics.mean(
            [product.time_in_system for product in self.customer_local.entities_of_system] +
            [product.time_in_system for product in self.customer_export.entities_of_system])
        # Average international transport time
        average_international_transport_time = statistics.mean(
            [product.international_transport_time for product in self.customer_local.entities_of_system] +
            [product.international_transport_time for product in self.customer_export.entities_of_system])
        # Average wholesales distributor time
        average_wholesales_distributor_time = statistics.mean(
            [product.wholesales_distributor_time for product in self.customer_local.entities_of_system] +
            [product.wholesales_distributor_time for product in self.customer_export.entities_of_system])
        # Average quantity
        average_quantity = statistics.mean(self.supplier_brazil.list_quantity)

        outcomes = {"Time_In_System": average_product_time_in_system,
                    "International_Transport_Time": average_international_transport_time,
                    "Wholesales_Time": average_wholesales_distributor_time,
                    "Quantity": average_quantity}

        ## Time series as dictionary
        # time_series = {"Supplier": self.supplier_brazil.hour_stats,
        #                "Manufacturer": self.manufacturer_brazil.hour_stats,
        #                "Exporter_Brazil": self.exporter_brazil.hour_stats,
        #                "Importer_Rotterdam": self.importer_rotterdam.hour_stats,
        #                "Importer_Senegal": self.importer_senegal.hour_stats,
        #                "Wholesales_Distributor": self.wholesales_distributor_eindhoven.hour_stats,
        #                "Retailer_Amsterdam": self.retailer_amsterdam.hour_stats,
        #                "Retailer_Utrecht": self.retailer_utrecht.hour_stats,
        #                "Retailer_Nijmegen": self.retailer_nijmegen.hour_stats,
        #                "Customer_Local": self.customer_local.hour_stats,
        #                "Customer_Export": self.customer_export.hour_stats}

        time_series = pd.DataFrame({"Supplier": pd.Series(self.supplier_brazil.hour_stats),
                                    "Manufacturer": pd.Series(self.manufacturer_brazil.hour_stats),
                                    "Exporter_Brazil": pd.Series(self.exporter_brazil.hour_stats),
                                    "Importer_Rotterdam": pd.Series(self.importer_rotterdam.hour_stats),
                                    "Importer_Senegal": pd.Series(self.importer_senegal.hour_stats),
                                    "Wholesales_Distributor": pd.Series(self.wholesales_distributor_eindhoven.hour_stats),
                                    "Retailer_Amsterdam": pd.Series(self.retailer_amsterdam.hour_stats),
                                    "Retailer_Utrecht": pd.Series(self.retailer_utrecht.hour_stats),
                                    "Retailer_Nijmegen": pd.Series(self.retailer_nijmegen.hour_stats),
                                    "Customer_Local": pd.Series(self.customer_local.hour_stats),
                                    "Customer_Export": pd.Series(self.customer_export.hour_stats)})
        time_series.reset_index(inplace=True)
        time_series = time_series.rename(columns={"index":"Time"})

        result = {"outcomes": outcomes, "time_series": time_series}

        return result
