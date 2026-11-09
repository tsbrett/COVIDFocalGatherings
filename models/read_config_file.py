import yaml
import pandas as pd
from models.extract_csv import read_datafiles
import numpy as np
import torch

def read_config_yaml(yaml_filepath="config.yml"):
    with open(yaml_filepath, 'r') as stream:
        try:
            config = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)
        return config


def get_model_tensors(data_dict, origin_date,  fips_codes=None):
    """
    TODO: Add functionality for fips_codes=None (simulate full network -- or at
     least full contiguous US)
    :param data_dict:
    :param fips_codes:
    :return:
    """
    df_local = data_dict["local_movement_data"]
    df_travel_matrices = data_dict["monthly_travel_matrices"]
    df_pop = data_dict["population_data"]

    #Note: not currently robust if user wants to simulate whole country
    df_local = (df_local[df_local["county"].isin(fips_codes)]
                .reset_index()
                .set_index(["county", "month"])
                ["pr_local_scaled"]
                .unstack(level=0))

    df_pop = df_pop[df_pop["fips"].isin(fips_codes)].reset_index()
    df_travel_matrices = df_travel_matrices.loc[df_travel_matrices.index
                                                .get_level_values(level="fips")
                                                .isin(fips_codes),
                                                df_travel_matrices.columns
                                                .isin(fips_codes)]

    # Start date of each month
    month_start_dates = pd.to_datetime(df_travel_matrices.index
                                       .get_level_values("month").unique()
                                       .to_series())
    # Start day for each month
    month_start_tensor = torch.tensor((month_start_dates
                                       - origin_date).dt.days.values)

    N_vector = torch.tensor(df_pop["pop"].values)  # Population by county

    n_counties = N_vector.shape[0]  # Total number of counties

    # Local activity tensor
    # local_tensor = torch.tensor(np.array([g["pr_local_scaled"] for _, g in
    #                                       df_local.groupby("month")]))
    local_tensor = torch.tensor(df_local.values)

    # Spatial movement tensor
    M_out_tensor = torch.tensor(np.array([g for _, g in
                                          df_travel_matrices.groupby("month")]))

    unscaled_M_back_tensor = (torch.ones(size=(n_counties, n_counties))
                              - torch.eye(n_counties))

    return dict(local_tensor=local_tensor,
                M_out_tensor=M_out_tensor,
                unscaled_M_back_tensor=unscaled_M_back_tensor,
                population_size_tensor=N_vector,
                month_start_tensor=month_start_tensor,
                location_names=df_pop[["fips", "name"]])


# Simulation parameters (used in defining model object)
config = read_config_yaml(yaml_filepath="config.yml")
data_dict = read_datafiles()

simulation_parameters = config["simulation_parameters"]


fips_codes = np.array(simulation_parameters["network_fips_codes"])
origin_date = pd.to_datetime(simulation_parameters["origin_date"])


model_tensors_dict = get_model_tensors(data_dict=data_dict,
                                       origin_date=origin_date,
                                       fips_codes=fips_codes)



p = simulation_parameters["t_eval"]
t_eval_date = pd.date_range(p["start_date"], p["end_date"], freq=p["freq"])
t_eval_day = (t_eval_date - origin_date).days  # Convert datetime to
# integer days relative to origin

os_parameters = simulation_parameters["outbreak_seeding_parameters"]
seed_fips_codes = os_parameters["seed_fips_codes"]
seed_counties = (model_tensors_dict["location_names"]
                 .reset_index()
                 .set_index("fips")
                 .loc[seed_fips_codes, "index"]
                 .values)  # Convert to index
seed_day = (pd.to_datetime(os_parameters["seed_date"]) - origin_date).days

# The day fixed_reduction_coef's reduction starts
sd_parameters = simulation_parameters["social_distancing_parameters"]
reduction_start_day = (pd.to_datetime(
    sd_parameters["social_distancing_start_date"]) - origin_date).days

reporting_parameters = simulation_parameters["reporting_parameters"]

# Epidemiological parameters (used to define parameters for simulation/inference run)

par_dict = config["default_model_parameters"]

# Iterated filtering: each parameter's random-walk sd (0 where it does not
# walk), in par_dict's order, and the sd's cooling
if_parameters = config["iterated_filtering_parameters"]
unknown = set(if_parameters["par_sd"]) - set(par_dict)
if unknown:
    raise KeyError(f"par_sd names no model parameter: {sorted(unknown)}")
rw_sd = [float(if_parameters["par_sd"].get(name, 0.)) for name in par_dict]

fixed_par_dict = dict(timestep=simulation_parameters["timestep"],
                      seed_day=seed_day,
                      seed_size=os_parameters["seed_size"],
                      seed_counties=seed_counties,
                      t_eval=t_eval_day,
                      t0=simulation_parameters["start_day"],
                      tf=simulation_parameters["end_day"],
                      reduction_start_day=reduction_start_day,
                      death_delay_weeks=reporting_parameters[
                          "death_delay_weeks"],
                      mean_floor=float(reporting_parameters["mean_floor"]),
                      rw_sd=rw_sd,
                      rw_alpha=float(if_parameters["alpha"]),
                      )



# Test:

# lmd = data_dict["local_movement_data"]
# lmd_unstack = (lmd[lmd["county"].isin(fips_codes)]
#                .groupby("county")["pr_local_scaled"]
#                .apply(lambda x: x)
#                .unstack(level=0))
#
# model_tensors_dict["local_tensor"]