
import collections
import numpy as np
import math
import random
import scipy.stats as stats



def handle_bounds(proposal):
    
    
#     # use folding, torroid or high dimensional donut approach
#     # or whatever you wish to call it to avoid going out of bounds
#     lower_bound_violation = ((new + delta) <= 0)
#     upper_bound_violation = ((new + delta) >= 1)
#     modified_delta = 1 - np.abs(delta)
#     delta[lower_bound_violation] = modified_delta[lower_bound_violation]
#     delta[upper_bound_violation] = -1*modified_delta[upper_bound_violation]
    
    # use reflection to deal with going out of bounds
    # modeled after dream
    lowerbound_violation = proposal < 0
    upperbound_violation = proposal > 1

    # TODO:: implement both folding and reflection as options
    if lowerbound_violation.any():
        violation_distance = 0 - proposal[lowerbound_violation]
        proposal[lowerbound_violation] = violation_distance
    if upperbound_violation.any():
        violation_distance = 1 - proposal[upperbound_violation]  
        proposal[upperbound_violation] = 1 + violation_distance
    
    if np.any((proposal < 0) | (proposal > 1)):
        l = (proposal <= 0) | (proposal  >= 0)
        proposal[l] = np.random.rand(np.sum(l))
        
    return proposal


def adaptive_metropolis(likelihood_func, bounds, n_draws):
    '''
    rough first implementation of adaptive metropolis following
    algorithm 2  matlab code in Vrugt (2015) for scenenario discovery use.
    
    this is based on the idea of approximate Bayesian computing and the code
    below follows DREAM-ABC for the handling of acceptance and rejection
    
    '''

    #Large number
    T=1e50

    d = bounds.shape[0]
    draws = np.empty((n_draws, d))
    scores = np.ones((n_draws,))*T
    rests = []
    n_accepted = 0
    cov = 2.38**2/d * np.eye(d)


    proposal = np.random.rand(d)
    rescaled = bounds[:, 0] + proposal * (bounds[:, 1]- bounds[:, 0])
    score = likelihood_func(rescaled)
    
    draws[0] = proposal
    scores[0] = score
    #rests.append(rest)
    
    for i in range(n_draws):
        if (i % 10 == 0) & (i > 20): # 20 is a bit arbitrary but you want some variant in draws 
            cov = 2.38**2/d * np.cov(draws[0:i, :], rowvar=False) + (1e-4 * np.eye(d))

        new_proposal = proposal.copy()  
        delta = np.random.multivariate_normal(np.zeros(d,), cov)
        new_proposal += delta
        new_proposal = handle_bounds(new_proposal)
               
        rescaled = bounds[:, 0] + new_proposal * (bounds[:, 1]- bounds[:, 0])
        score = likelihood_func(rescaled)

        if (score <= scores[i-1]) | (score == 0):
            scores[i] = score
            draws[i] = new_proposal
            proposal = new_proposal
            #rests.append(rest)
            n_accepted += 1
        else:
            scores[i] = scores[i-1]
            draws[i] = draws[i-1]
            #rests.append(rests[-1])

    # print(f"Acceptance percentage is {100*n_accepted/n_draws}%")
    
    mc = bounds[:, 0][:, np.newaxis] + draws.T * (bounds[:, 1]- bounds[:, 0])[:, np.newaxis]
    
    return mc.T, scores, (n_accepted/n_draws)
   
     
def differential_evolution_mc(likelihood_func, bounds, n_chains, n_draws):
    '''
    rough first implementation of differential evolution MC following
    algorithm 4  matlab code in Vrugt (2015) for scenario discovery use.
    
    this is based on the idea of approximate Bayesian computing and the code
    below follows DREAM-ABC for the handling of acceptance and rejection
    
    '''
    assert n_chains > 2
    
    d = bounds.shape[0] # number of uncertain dimensions
    
    gamma_RWM = 2.38/math.sqrt(2*d)
    draws = np.empty((n_draws, d, n_chains))
    scores = np.empty((n_draws, n_chains))
    #rests = collections.defaultdict(list)
    standard_normal = stats.norm()
    n_accepted = 0
    
    proposals = np.random.rand(n_chains, d)
    
    for chain_i in range(n_chains):
        proposal = proposals[chain_i, :].copy()
        rescaled = bounds[:, 0] + proposal * (bounds[:, 1]- bounds[:, 0])
        score = likelihood_func(rescaled)
        draws[0, :, chain_i] = proposal
        scores[0, chain_i] = score
        #rests[chain_i].append(rest)
        # TODO: we need to store score also on a chain per chain basis
        # might be idea to have chains as small objects with state
        # will make some form of parallelization easier
    
    R = np.asarray([[x for x in range(n_chains) if x!= i] for i in range(n_chains) ])
    
    for draw_i in range(1, n_draws):
        
        g = random.choices([gamma_RWM, 1], weights=[0.9, 0.1], k=1) #crossover
        draw = np.argsort(np.random.rand(n_chains-1, n_chains), axis=0)
        for chain_i in range(n_chains):
            a = R[chain_i, draw[0, chain_i]]
            b = R[chain_i, draw[1, chain_i]]
            
            current = proposals[chain_i, :].copy()
            
            candidate_a = draws[draw_i-1, :, a]
            candidate_b = draws[draw_i-1, :,b]
            
            new_proposal = current + g * (candidate_a - candidate_b) +\
                            1e-6*standard_normal.rvs(d) 
            new_proposal = handle_bounds(new_proposal)
                        
            rescaled = bounds[:, 0] + new_proposal * (bounds[:, 1]- bounds[:, 0])
            score = likelihood_func(rescaled)
            
            if (score <= scores[draw_i-1, chain_i ]) | (score == 0):
                proposals[chain_i, :] = new_proposal
                draws[draw_i, :, chain_i] = new_proposal
                scores[draw_i, chain_i] = score
                #rests[chain_i].append(rest)
                n_accepted += 1
            else:
                draws[draw_i, :, chain_i] = draws[draw_i-1, :, chain_i]
                scores[draw_i, chain_i] = scores[draw_i-1, chain_i]
                #rests[chain_i].append(rests[chain_i][-1])

    
    # print(f"acceptance percentage is {100*n_accepted/(n_chains*n_draws)}%")

    mc = bounds[:, 0][:, np.newaxis] + draws * (bounds[:, 1]- bounds[:, 0])[:, np.newaxis]

    return mc, scores, (n_accepted/(n_chains*n_draws))


def dream(likelihood_func, bounds, n_chains, n_draws, n_burnin=250):
    '''
    rough first implementation of DREAM following algorithm 5  matlab code in
    Vrugt (2015) for scenario discovery use.
    
    this code does include a burnin period for adapting the crossover
    probabilities. It does not include the outlier detecting correction through
    the check function as discussed by Vrugt (penultimate line of code in 
    algorithm 5). 
    
    this is based on the idea of approximate Bayesian computing and the code
    below follows DREAM-ABC for the handling of acceptance and rejection
    
    '''
    delta = 3
    c = 0.1
    c_star = 1e-12
    n_crossover = 3 # number of dimensions to consider in crossovers
    p_gamma = 0.2
    d = bounds.shape[0] # number of uncertain dimensions
    n_accepted = 0
    J = np.zeros(n_crossover)
    n_id = np.zeros(n_crossover)
    
    draws = np.empty((n_draws, d, n_chains))
    scores = np.empty((n_draws, n_chains))
    #rests = collections.defaultdict(list)
    R = np.asarray([[x for x in range(n_chains) if x!= i] for i in range(n_chains) ])
    proposals = np.random.rand(n_chains, d)

    # first iteration
    for chain_i in range(n_chains):
        proposal = proposals[chain_i, :].copy()
        rescaled = bounds[:, 0] + proposal * (bounds[:, 1]- bounds[:, 0])
        score = likelihood_func(rescaled)
        draws[0, :, chain_i] = proposal
        scores[0, chain_i] = score
        #rests[chain_i].append(rest)
    
    for draw_i in range(1, n_draws):        
        draw = np.argsort(np.random.rand(n_chains-1, n_chains), axis=0)
        lambda_for_chains = -c + 2*c*np.random.rand(n_chains) 
        std_X = np.std(proposals, axis=0)

        for chain_i in range(n_chains):
            D = np.random.randint(1, delta+1)
            a = R[chain_i, draw[0:D, chain_i]]
            b = R[chain_i, draw[D:2*D, chain_i]]   # -> sample from past states
            
            n_crossoverdims = np.random.choice(n_crossover, (1,))+1
            dimensions = np.random.choice(d, (n_crossoverdims[0],),
                                          replace=False) 
            d_star = dimensions.shape[0]

            gamma_d = 2.38 / math.sqrt(2*d*d_star)
            g = random.choices([gamma_d, 1], weights=[1-p_gamma, p_gamma], k=1)[0]
            
            a_proposals = proposals[a[:, None], dimensions]
            b_proposals = proposals[b[:, None], dimensions]
            delta_proposals = np.sum(a_proposals, axis=0)-np.sum(b_proposals, axis=0)
            
            d_p = c_star*np.random.randn(d_star) +\
                  lambda_for_chains[chain_i]*g*np.sum(delta_proposals, axis=0)
                  
            new_proposal = proposals[chain_i,:].copy()
            new_proposal[dimensions] += d_p
            new_proposal = handle_bounds(new_proposal)
                        
            rescaled = bounds[:, 0] + new_proposal * (bounds[:, 1]- bounds[:, 0])
            score = likelihood_func(rescaled)
            
            if (score <= scores[draw_i-1, chain_i ]) | (score == 0):
                proposals[chain_i, :] = new_proposal[:]
                draws[draw_i, :, chain_i] = new_proposal[:]
                scores[draw_i, chain_i] = score
                #rests[chain_i].append(rest)
                n_accepted += 1
            else:
                draws[draw_i, :, chain_i] = draws[draw_i-1, :, chain_i]
                scores[draw_i, chain_i] = scores[draw_i-1, chain_i]
                #rests[chain_i].append(rests[chain_i][-1])
            
            J[n_crossoverdims-1] += np.sum((d_p / std_X[dimensions])**2)
            n_id[n_crossoverdims-1] += 1
        
        if draw_i % 10 == 0 and n_draws<n_burnin:
            p_CR = J / n_id
            p_CR = p_CR / np.sum(p_CR) 
            
    # print(f"Acceptance percentage is {100*n_accepted/(n_chains*n_draws)}%")

    mc = bounds[:, 0][:, np.newaxis] + draws * (bounds[:, 1]- bounds[:, 0])[:, np.newaxis]

    return mc, scores, (n_accepted/(n_chains*n_draws))
             
if __name__ == '__main__':
    def calculate_likelihood(params):
        a = random.randint(0,30)
        # Calculate distance metrics
        dist = 0
        for par in params:
            dist += (a - par)
        print("Parameter Vector", params, "Dist", dist)
        return dist
    
    bounds = np.asarray([[-0.5, 0.5], [-0.5, 1], [-0.5, 1]])
    n_burnin = 100
    results, scores, acceptance_percentage = adaptive_metropolis(calculate_likelihood, bounds, 1000)
    #results, scores, acceptance_percentage = dream(calculate_likelihood, bounds, 6, 5000, n_burnin=n_burnin)
    #results, scores, acceptance_percentage = differential_evolution_mc(calculate_likelihood, bounds, 3, 1000)