from pydream.core import run_dream
from pydream.parameters import SampledParam
from scipy.stats import uniform

from police_simulation_model.run_police_simulation_model import (
    PoliceSimModel,
)


def likelihood_distance(parameter_vector):
    #Run simulation
    par1 = parameter_vector[0]
    par2 = round(parameter_vector[1])

    result_sim_model = PoliceSimModel.run(parameters=[par1, par2])

    a=5
    #Calculate distance metrics
    dist = 0
    for par in parameter_vector:
        dist += (a-par)
    print("Parameter Vector", parameter_vector, "Dist", dist)
    return dist


if __name__ == '__main__':
    #loc is lower bound, scale is upper bound minus lower bound
    #len of list = number of decision variables


    parameters_to_sample = SampledParam(
        uniform, loc=[1, 0], scale=[3, 1])

    print("Parameters are sampled")

    sampled_params, log_ps = run_dream([parameters_to_sample], likelihood_distance, nchains=3,
                                       niterations=5, multiprocessing=False)

    print("ABC is done")