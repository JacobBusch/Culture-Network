"""Runs a single simulation to produce data which is saved

A module that use dictionary of data for the simulation run. The single shot simulztion is run
for a given initial set seed.



Created: 10/10/2022
"""
# imports
from package.resources.run_MC import generate_data
from package.resources.utility import (
    createFolder, 
    save_object, 
    produce_name_datetime
)
import json

def main(
    base_params,
    return_data=False
) -> str: 

    root = "single_experiment"
    fileName = produce_name_datetime(root)
    print("fileName:", fileName)

    Data = generate_data(base_params)  # run the simulation

    createFolder(fileName)
    save_object(Data, fileName + "/Data", "social_network")
    save_object(base_params, fileName + "/Data", "base_params")
    
    if return_data:
    # Beispiel: durchschnittlicher Threshold-Verlauf
        return fileName, {
            "av_thresholds_list": Data.history_av_thresholds,
            "av_identity_list":  Data.history_average_identity,
            "av_carbon_emission": Data.history_total_carbon_emissions_flow            
    }
    else:
        return fileName

    

if __name__ == '__main__':
    BASE_PARAMS_LOAD = "package/constants/base_params_single_run.json"
    # load base params
    f = open(BASE_PARAMS_LOAD)
    base_params = json.load(f)

    fileName = main(
        base_params = base_params
    )