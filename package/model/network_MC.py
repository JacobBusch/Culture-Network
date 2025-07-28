"""Create social network with individuals
A module that use input data to generate a social network containing individuals who each have multiple 
behaviours. The weighting of individuals within the social network is determined by the identity distance 
between neighbours. The simulation evolves over time saving data at set intervals to reduce data output.


Created: 10/10/2022
"""

# imports
import numpy as np
import networkx as nx
import numpy.typing as npt
from package.model.individuals_MC import Individual
from package.model.one_m_green_influencer import Individual_one_m_green_influencer

# modules
class Network:
    """
    Class to represent social network of simulation which is composed of individuals each with identities and behaviours

    ...

    Parameters
    ----------
    parameters : dict
        Dictionary of parameters used to generate attributes, dict used for readability instead of super long list of input parameters

    Attributes
    ----------
    set_seed : int
        stochastic seed of simulation for reproducibility
    alpha_change : char
        determines how  and how often agent's re-asses their connections strength in the social network
    save_timeseries_data : bool
        whether or not to save data. Set to 0 if only interested in end state of the simulation.
    compression_factor: int
        how often data is saved. If set to 1 its every step, then 10 is every 10th steps. Higher value gives lower
        resolution for graphs but more managable saved or end object size
    t: float
        keep track of time
    M: int
        number of behaviours per individual. These behaviours are meant to represent action decisions that operate under the
        same identity such as the decision to cycle to work or take the car.
    N: int
        number of individuals in the social network
    K: int
        number of connections per individual in the small world social network. The greater the prob_rewire the greater
        agent to agent variation there is in this number. Must be less than N.
    prob_rewire: float
        Probability of rewiring connections in the social network from one indivdual to another. The greater this values
        the more long distance connections within the network exist. Domain = [0,1]
    cultural_inertia: float
        the number of steps into the past that are considered when individuals consider their identity
    discount_factor: float
        the degree to which each previous time step has a decreasing importance to an individuals memory. Domain = [0,1]
    normalized_discount_array: npt.NDArray[float]
        discounting time series that is the length of that agents cultural_inertia. The array is row normalized so that each row sums to 1
    confirmation_bias: float
        the extent to which individuals will only pay attention to other idividuals who are similar to them in social interactions
    learning_error_scale: float
        the standard deviation of a guassian distribution centered on zero, representing the imperfection of learning in social transmission
    phi_array: npt.NDArray[float]
        list of degree of social susceptibility or conspicous consumption of the different behaviours. 
    homophily: float
        the degree to which an agents neighbours are similar to them identity wise. A value of 1 means agents are placed in the small world social network
        next to their cultural peers. The closer to 0 the more times agents are swapped around in the network using the Fisher Yates shuffle. Domain [0,1]
    shuffle_reps: int
        the number of time indivdiuals swap positions within the social network, note that this doesn't affect the actual network structure
    a_attitude,bb_attitude, a_threshold, b_threshold: float
        parameters used to generate the beta distribution for the initial agent attitudes and thresholds for each behaviour respectively.
        The same distribution is used for all agents and behaviours
    attitude_matrix_init: npt.NDArray[float]
        array of shape (N,M) with the initial values of the attitudes of each agent towards M behaviours which then evolve over time. Used to generated
        Indivdual objects
    threshold_matrix_init: npt.NDArray[float]
        array of shape (N,M) with the initial values of the thresholds of each agent towards M behaviours, these are static. Used to generated
        Indivdual objects
    agent_list: list[Individual]
        list of Individuals objects containing behaviours of each individual
    adjacency_matrix: npt.NDArray[bool]
        array giveing social network structure where 1 represents a connection between agents and 0 no connection. It is symetric about the diagonal
    weighting_matrix: npt.NDArray[float]
        an NxN array how how much each agent values the opinion of their neighbour. Note that is it not symetric and agent i doesn't need to value the
        opinion of agent j as much as j does i's opinion
    network: nx.Graph
        a networkx watts strogatz small world graph
    social_component_matrix: npt.NDArray[float]
        NxM array of influence of neighbours on an individual's attitudes towards M behaviours
    average_identity: float
        average identity of society
    std_identity : float
        standard deviation of agent identies at time t
    var_identity: float
        variance of agent identies at time t
    min_identity: float
        minimum individual identity at time t
    max_identity: float
        maximum individual idenity at time t
    weighting_matrix_convergence: float
        total change in agent link strength from previous to current step, a measure of convergece should tend to zero
    total_carbon_emissions: float
        total emissions due to behavioural choices of agents. Note the difference between this and carbon_emissions list for each behaviour
    history_weighting_matrix: list[npt.NDArray[float]]
        time series of weighting_matrix
    history_social_component_matrix: list[npt.NDArray[float]]
        time series of social_component_matrix
    history_var_identity: list[float]
        time series of var_identity
    history_time: list[float]
        time series of time
    history_total_carbon_emission: list[float]
        time series of total_carbon_emissions in the system, not the carbon_emissions for each behaviour
    history_weighting_matrix_convergence: list[float]
        time series of weighting_matrix_convergence
    history_average_identity: list[float]
        time series of average agent identity
    history_std_identity: list[float]
        time series of std_identity
    history_min_identity: list[float]
        time series of min_identity
    

    Methods
    -------
    normlize_matrix(matrix: npt.NDArray) ->  npt.NDArray:
        Row normalize an array
    calc_normalized_discount_array(self):
        Returns row normalized discount array
    create_weighting_matrix()-> tuple[npt.NDArray, npt.NDArray, nx.Graph]:
        Create small world social network
    produce_circular_list(list) -> list:
        Makes an ordered list circular so that the start and end values are close in value
    partial_shuffle(l, swap_reps) -> list:
        Partially shuffle a list using Fisher Yates shuffle
    generate_init_data_behaviours() -> tuple:
        Generate the initial values for agent behavioural attitudes and thresholds
    create_agent_list() -> list:
        Create list of Individual objects that each have behaviours
    calc_ego_influence_degroot() ->  npt.NDArray:
        Calculate the influence of neighbours using the Degroot model of weighted aggregation
    calc_social_component_matrix() ->  npt.NDArray:
        Combine neighbour influence and social learning error to updated individual behavioural attitudes
    calc_total_weighting_matrix_difference(matrix_before: npt.NDArray, matrix_after: npt.NDArray)-> float:
        Calculate the total change in link strength over one time step
    update_weightings()-> float:
        Update the link strength array according to the new agent identities
    calc_total_emissions() -> int:
        Calculate total carbon emissions of N*M behaviours
    calc_network_identity() ->  tuple[float, float, float, float]:
        Return various identity properties
    update_individuals():
        Update Individual objects with new information
    save_timeseries_data_network():
        Save time series data
    next_step():
        Push the simulation forwards one time step
    """

    def __init__(self, parameters: dict):
        """
        Constructs all the necessary attributes for the Network object.

        Parameters
        ----------
        parameters : dict
            Dictionary of parameters used to generate attributes, dict used for readability instead of super long list of input parameters

        """

        self.set_seed = parameters["set_seed"]
        np.random.seed(self.set_seed)

        self.K = int(round(parameters["K"]))  # round due to the sampling method producing floats in the Sobol Sensitivity Analysis
        self.prob_rewire = parameters["prob_rewire"]
        self.alpha_change = parameters["alpha_change"]
        self.save_timeseries_data = parameters["save_timeseries_data"]
        self.compression_factor = parameters["compression_factor"]

        # time
        self.t = 0

        # network
        self.green_N = int(round(parameters["green_N"]))
        self.M = int(round(parameters["M"]))
        self.N = int(round(parameters["N"]))
        self.Model_Setup = parameters["Model_Setup"]
        self.time_steps_max = parameters["time_steps_max"]
        
        # identity
        self.cultural_inertia = int(round(parameters["cultural_inertia"]))

        # time discounting
        self.discount_factor = parameters["discount_factor"]
        self.normalized_discount_array = self.calc_normalized_discount_array()

        # social learning and bias
        self.confirmation_bias = parameters["confirmation_bias"]
        self.learning_error_scale = parameters["learning_error_scale"]

        # social influence of behaviours
        self.phi_lower = parameters["phi_lower"]
        self.phi_upper = parameters["phi_upper"]
        self.phi_array = np.linspace(self.phi_lower, self.phi_upper, num=self.M)

        # network homophily
        self.homophily = parameters["homophily"]  # 0-1
        self.shuffle_reps = int(
            round(self.N*(1 - self.homophily))
        )

        # create network
        if self.green_N>0:
            (
                self.adjacency_matrix,
                self.weighting_matrix,
                self.network,
            ) = self.create_weighting_matrix_add_greens()
        else:
            (
                self.adjacency_matrix,
                self.weighting_matrix,
                self.network,
            ) = self.create_weighting_matrix()

        if self.alpha_change == "behavioural_independence":
            self.weighting_matrix_list = [self.weighting_matrix]*self.M

        self.network_density = nx.density(self.network)
        
        self.a_attitude = parameters["a_attitude"]
        self.b_attitude = parameters["b_attitude"]
        self.a_threshold = parameters["a_threshold"]
        self.b_threshold = parameters["b_threshold"]
        self.a_pU =  np.random.uniform(0.1, 15)
        self.a_pC =  np.random.uniform(0.1, 15)
        self.a_pR =  np.random.uniform(0.1, 15)
        self.b_pU =  np.random.uniform(0.1, 15)
        self.b_pC =  np.random.uniform(0.1, 15)
        self.b_pR =  np.random.uniform(0.1, 15)
        self.a_thresholdpU =  np.random.uniform(0.1, 15)
        self.a_thresholdpC =  np.random.uniform(0.1, 15)
        self.a_thresholdpR =  np.random.uniform(0.1, 15)
        self.b_thresholdpU =  np.random.uniform(0.1, 15)
        self.b_thresholdpC =  np.random.uniform(0.1, 15)
        self.b_thresholdpR =  np.random.uniform(0.1, 15)

        (
            self.attitude_matrix_init,
            self.threshold_matrix_init,
            self.thresholdspU_init,
            self.thresholdspC_init,
            self.thresholdspR_init,
            self.pU_matrix_init,
            self.pC_matrix_init,
            self.pR_matrix_init
        ) = self.generate_init_data_behaviours()

        self.agent_list = self.create_agent_list()

        if self.green_N > 0:
            self.add_green_influencers_list()
            self.N = len(self.agent_list)

        self.shuffle_agent_list()#partial shuffle of the list based on identity

        self.social_component_matrix = self.calc_social_component_matrix()
        self.TA_component_matrix = self.calc_TA_component_matrix()

        if self.alpha_change == ("static_culturally_determined_weights" or "dynamic_culturally_determined_weights"):
            self.weighting_matrix, self.total_identity_differences,__ = self.update_weightings()
        elif self.alpha_change == "behavioural_independence":#independent behaviours
            self.weighting_matrix_list = self.update_weightings_list()

        self.init_total_carbon_emissions  = self.calc_total_emissions_flow()
        self.total_carbon_emissions_flow = self.init_total_carbon_emissions
        self.total_carbon_emissions_stock = self.init_total_carbon_emissions

        (
                self.identity_list,
                self.average_identity,
                self.std_identity,
                self.var_identity,
                self.min_identity,
                self.max_identity,
        ) = self.calc_network_identity()

        self.av_behaviour = np.mean([np.mean(a.attitudes) for a in self.agent_list])
        self.av_thresholds = np.mean([agent.av_thresholds for agent in self.agent_list])

        
        if self.save_timeseries_data:
            self.history_weighting_matrix = [self.weighting_matrix]
            self.history_social_component_matrix = [self.social_component_matrix]
            self.history_social_TA_matrix = [self.TA_component_matrix]
            self.history_time = [self.t]
            self.weighting_matrix_convergence = 0  # there is no convergence in the first step, to deal with time issues when plotting
            self.history_weighting_matrix_convergence = [
                self.weighting_matrix_convergence
            ]
            self.history_average_identity = [self.average_identity]
            self.history_av_behaviour = [self.av_behaviour]
            self.history_av_thresholds = [self.av_thresholds]
            self.history_std_identity = [self.std_identity]
            self.history_var_identity = [self.var_identity]
            self.history_min_identity = [self.min_identity]
            self.history_max_identity = [self.max_identity]
            self.history_total_carbon_emissions_flow = [self.total_carbon_emissions_flow]
            self.history_total_carbon_emissions_stock = [self.total_carbon_emissions_stock]
            if self.alpha_change == ("static_culturally_determined_weights" or "dynamic_culturally_determined_weights"):
                self.history_total_identity_differences = [self.total_identity_differences]

    def normlize_matrix(self, matrix: npt.NDArray) -> npt.NDArray:
        """
        Row normalize an array

        Parameters
        ----------
        matrix: npt.NDArrayf
            array to be row normalized

        Returns
        -------
        norm_matrix: npt.NDArray
            row normalized array
        """
        row_sums = matrix.sum(axis=1)
        norm_matrix = matrix / row_sums[:, np.newaxis]

        return norm_matrix

    def calc_normalized_discount_array(self) -> npt.NDArray:
        """
        produce normalized discount array

        Parameters
        ----------
        None

        Returns
        -------
        normalized_discount_array: npt.NDArray
            row normalized truncated quasi-hyperbolic discount array
        """

        discount_row = [(self.discount_factor)**(v) for v in range(self.cultural_inertia)]
        normalized_discount_array = (np.asarray(discount_row)/sum(discount_row))


        return normalized_discount_array 

    def create_weighting_matrix(self) -> tuple[npt.NDArray, npt.NDArray, nx.Graph]:
        """
        Create watts-strogatz small world graph using Networkx library

        Parameters
        ----------
        None

        Returns
        -------
        weighting_matrix: npt.NDArray[bool]
            adjacency matrix, array giving social network structure where 1 represents a connection between agents and 0 no connection. It is symetric about the diagonal
        norm_weighting_matrix: npt.NDArray[float]
            an NxN array how how much each agent values the opinion of their neighbour. Note that is it not symetric and agent i doesn't need to value the
            opinion of agent j as much as j does i's opinion
        ws: nx.Graph
            a networkx watts strogatz small world graph
        """

        G = nx.watts_strogatz_graph(n=self.N, k=self.K, p=self.prob_rewire, seed=self.set_seed)

        weighting_matrix = nx.to_numpy_array(G)

        norm_weighting_matrix = self.normlize_matrix(weighting_matrix)

        return (
            weighting_matrix,
            norm_weighting_matrix,
            G,
        )
    
    def create_weighting_matrix_add_greens(self):
        """
        Create watts-strogatz small world graph using Networkx library but with N GREEN indivduals

        Parameters
        ----------
        None

        Returns
        -------
        weighting_matrix: npt.NDArray[bool]
            adjacency matrix, array giving social network structure where 1 represents a connection between agents and 0 no connection. It is symetric about the diagonal
        norm_weighting_matrix: npt.NDArray[float]
            an NxN array how how much each agent values the opinion of their neighbour. Note that is it not symetric and agent i doesn't need to value the
            opinion of agent j as much as j does i's opinion
        ws: nx.Graph
            a networkx watts strogatz small world graph
        """

        G = nx.watts_strogatz_graph(n=self.N+self.green_N, k=self.K, p=self.prob_rewire, seed=self.set_seed)

        weighting_matrix = nx.to_numpy_array(G)

        norm_weighting_matrix = self.normlize_matrix(weighting_matrix)

        return (
            weighting_matrix,
            norm_weighting_matrix,
            G,
        )


    def circular_agent_list(self) -> list:
        """
        Makes an ordered list circular so that the start and end values are matched in value and value distribution is symmetric

        Parameters
        ----------
        list: list
            an ordered list e.g [1,2,3,4,5]
        Returns
        -------
        circular: list
            a circular list symmetric about its middle entry e.g [1,3,5,4,2]
        """

        first_half = self.agent_list[::2]  # take every second element in the list, even indicies
        second_half = (self.agent_list[1::2])[::-1]  # take every second element , odd indicies
        self.agent_list = first_half + second_half

    def partial_shuffle_agent_list(self) -> list:
        """
        Partially shuffle a list using Fisher Yates shuffle
        """

        for _ in range(self.shuffle_reps):
            a, b = np.random.randint(
                low=0, high=self.N, size=2
            )  # generate pair of indicies to swap
            self.agent_list[b], self.agent_list[a] = self.agent_list[a], self.agent_list[b]

    def generate_init_data_behaviours(self) -> tuple[npt.NDArray, npt.NDArray, npt.NDArray, npt.NDArray, npt.NDArray, npt.NDArray, npt.NDArray, npt.NDArray ]:
        """
        Generate the initial values for agent behavioural attitudes and thresholds using Beta distribution

        Parameters
        ----------
        None

        Returns
        -------
        attitude_matrix: npt.NDArray
            NxM array of behavioural attitudes
        threshold_matrix: npt.NDArray
            NxM array of behavioural thresholds, represents the barriers to entry to performing a behaviour e.g the distance of a
            commute or disposable income of an individual
        """

        attitude_list = [
            np.random.beta(self.a_attitude, self.b_attitude, size=self.M)
            for n in range(self.N)
        ]

        threshold_list = [
            np.random.beta(self.a_threshold, self.b_threshold, size=self.M)
            for n in range(self.N)
        ]

        pU_list = [
            np.random.beta(self.a_pU, self.b_pU, size=self.M)
            for n in range(self.N)
        ]

        pC_list = [
            np.random.beta(self.a_pC, self.b_pC, size=self.M)
            for n in range(self.N)
        ]

        pR_list = [
            np.random.beta(self.a_pR, self.b_pR, size=self.M)
            for n in range(self.N)
        ]

        thresholdspU_list = [
            np.random.beta(self.a_thresholdpU, self.b_thresholdpU, size=self.M)
            for n in range(self.N)
        ]

        thresholdspC_list = [
            np.random.beta(self.a_thresholdpC, self.b_thresholdpC, size=self.M)
            for n in range(self.N)
        ]

        thresholdspR_list = [
            np.random.beta(self.a_thresholdpR, self.b_thresholdpR, size=self.M)
            for n in range(self.N)
        ]


        attitude_matrix = np.asarray(attitude_list)
        threshold_matrix = np.asarray(threshold_list)
        pU_matrix = np.asarray(pU_list)
        pC_matrix = np.asarray(pC_list)
        pR_matrix = np.asarray(pR_list)
        thresholdspU_matrix = np.asarray(thresholdspU_list)
        thresholdspC_matrix = np.asarray(thresholdspC_list)
        thresholdspR_matrix = np.asarray(thresholdspR_list)



        return attitude_matrix, threshold_matrix, pU_matrix, pC_matrix, pR_matrix, thresholdspU_matrix, thresholdspC_matrix, thresholdspR_matrix

    def create_agent_list(self) -> list[Individual]:
        """
        Create list of Individual objects that each have behaviours

        Parameters
        ----------
        None

        Returns
        -------
        agent_list: list[Individual]
            List of Individual objects representing specific agents in a social network each with behavioural opinions and environmental identity
        """

        individual_params = {
            "t": self.t,
            "time_steps_max": self.time_steps_max,
            "M": self.M,
            "save_timeseries_data": self.save_timeseries_data,
            "phi_array": self.phi_array,
            "compression_factor": self.compression_factor,
            "alpha_change" : self.alpha_change,
            "Model_Setup" : self.Model_Setup,
        }

        agent_list = [
            Individual(
                individual_params,
                self.attitude_matrix_init[n],
                self.threshold_matrix_init[n],
                self.normalized_discount_array,
                self.cultural_inertia,
                self.pU_matrix_init[n],
                self.pC_matrix_init[n],
                self.pR_matrix_init[n],
                self.thresholdspU_init[n],
                self.thresholdspC_init[n],
                self.thresholdspR_init[n],
                n
            )
            for n in range(self.N)
        ]

        return agent_list
        
    def add_green_influencers_list(self):
        """Add green influencers to agent list"""

        attitude_list_green_N = [
            np.random.beta(self.a_attitude, self.b_attitude, size=self.M)
            for n in range(self.green_N)
        ]

        threshold_list_green_N  = [
            np.random.beta(self.a_threshold, self.b_threshold, size=self.M)
            for n in range(self.green_N)
        ]

        individual_params = {
            "t": self.t,
            "M": self.M,
            "save_timeseries_data": self.save_timeseries_data,
            "phi_array": self.phi_array,
            "compression_factor": self.compression_factor,
            "alpha_change" : self.alpha_change
        }

        agent_green_influencer_list = [
            Individual_one_m_green_influencer(
                individual_params,
                attitude_list_green_N[n],
                threshold_list_green_N[n],
                self.normalized_discount_array,
                self.cultural_inertia,
                self.N + n
            )
            for n in range(self.green_N)
        ]

        self.agent_list.extend(agent_green_influencer_list)


    def shuffle_agent_list(self): 
        #make list cirucalr then partial shuffle it
        self.agent_list.sort(key=lambda x: x.identity)#sorted by identity
        self.circular_agent_list()#agent list is now circular in terms of identity
        self.partial_shuffle_agent_list()#partial shuffle of the list

    def calc_ego_influence_degroot(self) -> npt.NDArray:
        """
        Calculate the influence of neighbours using the Degroot model of weighted aggregation

        Parameters
        ----------
        None

        Returns
        -------
        neighbour_influence: npt.NDArray
            NxM array where each row represents the influence of an Individual listening to its neighbours regarding their
            behavioural attitude opinions, this influence is weighted by the weighting_matrix
        """

        behavioural_attitude_matrix = np.asarray([n.attitudes for n in self.agent_list])
        neighbour_influence = np.matmul(self.weighting_matrix, behavioural_attitude_matrix)
        
        return neighbour_influence
    
    def calc_ego_influence_degroot_TA(self) -> tuple[npt.NDArray, npt.NDArray, npt.NDArray]:
        
        pU_matrix = np.asarray([n.pU for n in self.agent_list])
        neighbour_pU_influence = np.matmul(self.weighting_matrix, pU_matrix)
        
        pC_matrix = np.asarray([n.pC for n in self.agent_list])
        neighbour_pC_influence = np.matmul(self.weighting_matrix, pC_matrix)

        pR_matrix = np.asarray([n.pR for n in self.agent_list])
        neighbour_pR_influence = np.matmul(self.weighting_matrix, pR_matrix)

        return neighbour_pU_influence, neighbour_pC_influence, neighbour_pR_influence 

    def calc_ego_influence_degroot_independent(self) -> npt.NDArray:
        """
        Calculate the influence of neighbours using the Degroot model of weighted aggregation, BEHAVIOURS INDEPENDANT ("alpha_change" case D)

        Parameters
        ----------
        None

        Returns
        -------
        neighbour_influence: npt.NDArray
            NxM array where each row represents the influence of an Individual listening to its neighbours regarding their
            behavioural attitude opinions, this influence is weighted by the weighting_matrix
        """

        behavioural_attitude_matrix = np.asarray([n.attitudes for n in self.agent_list])
        neighbour_influence = np.zeros((self.N, self.M))

        for m in range(self.M):
            neighbour_influence[:, m] = np.matmul(self.weighting_matrix_list[m], behavioural_attitude_matrix[:,m])

        return neighbour_influence

    def calc_social_component_matrix(self) -> npt.NDArray:
        """
        Combine neighbour influence and social learning error to updated individual behavioural attitudes

        Parameters
        ----------
        None

        Returns
        -------
        social_influence: npt.NDArray
            NxM array giving the influence of social learning from neighbours for that time step
        """

        if self.alpha_change == "behavioural_independence":
            ego_influence = self.calc_ego_influence_degroot_independent()
        else:
            ego_influence = self.calc_ego_influence_degroot()           

        social_influence = ego_influence + np.random.normal(
            loc=0, scale=self.learning_error_scale, size=(self.N, self.M)
        )
        return social_influence

    def calc_TA_component_matrix(self) -> tuple[npt.NDArray, npt.NDArray, npt.NDArray]:

        if self.alpha_change == "behavioural_independence":
            raise NotImplementedError
        else:
            ego_TA_influence_pU, ego_TA_influence_pC, ego_TA_influence_pR, = self.calc_ego_influence_degroot_TA()           

        social_influence_pU = ego_TA_influence_pU + np.random.normal(
            loc=0, scale=self.learning_error_scale, size=(self.N, self.M)
        )

        social_influence_pC = ego_TA_influence_pC + np.random.normal(
            loc=0, scale=self.learning_error_scale, size=(self.N, self.M)
        )

        social_influence_pR = ego_TA_influence_pR + np.random.normal(
            loc=0, scale=self.learning_error_scale, size=(self.N, self.M)
        )

        return social_influence_pU, social_influence_pC, social_influence_pR

    def calc_total_weighting_matrix_difference(
        self, matrix_before: npt.NDArray, matrix_after: npt.NDArray
    ) -> float:
        """
        Calculate the total change in link strength over one time step. Meant to show how the weighting quickly converges

        Parameters
        ----------
        matrix_before: npt.NDArray
            NxN array of agent opinion weightings from the previous time step
        matrix_after: npt.NDArray
            NxN array of agent opinion weightings from the current time step

        Returns
        -------
        total_difference: float
            total element wise difference between the arrays
        """
        difference_matrix = np.subtract(matrix_before, matrix_after)
        total_difference = (np.abs(difference_matrix)).sum()
        return total_difference

    def update_weightings(self) -> tuple[npt.NDArray, float]:
        """
        Update the link strength array according to the new agent identities

        Parameters
        ----------
        None

        Returns
        -------
        norm_weighting_matrix: npt.NDArray
            Row normalized weighting array giving the strength of inter-Individual connections due to similarity in identity
        total_identity_differences
        total_difference: float
            total element wise difference between the previous weighting arrays
        """
        identity_list = np.array([x.identity for x in self.agent_list])

        difference_matrix = np.subtract.outer(identity_list, identity_list)

        alpha_numerator = np.exp(
            -np.multiply(self.confirmation_bias, np.abs(difference_matrix))
        )

        non_diagonal_weighting_matrix = (
            self.adjacency_matrix * alpha_numerator
        )  # We want only those values that have network connections

        norm_weighting_matrix = self.normlize_matrix(
            non_diagonal_weighting_matrix
        )  # normalize the matrix row wise

        #for total_identity_differences
        difference_matrix_real_connections = abs(self.adjacency_matrix * difference_matrix)
        total_identity_differences = difference_matrix_real_connections.sum(axis=1)

        if self.save_timeseries_data:
            total_difference = self.calc_total_weighting_matrix_difference(
                self.weighting_matrix, norm_weighting_matrix
            )
            return norm_weighting_matrix, total_identity_differences, total_difference 
        else:
            return norm_weighting_matrix, total_identity_differences, 0
    
    def update_weightings_list(self):
        """
        Update the link strength array according to the agent ATTITUDES NOT IDENTITIES, RETURN LIST OF MATRICIES

        Parameters
        ----------
        None

        Returns
        -------
        weighting_matrix_list: list[npt.NDArray]
            List of row normalized weighting array giving the strength of inter-Individual connections due to similarity in attitude
        """
        weighting_matrix_list = []

        for m in range(self.M):
            attitude_star_list = np.array([x.attitudes_star[m] for x in self.agent_list])

            difference_matrix = np.subtract.outer(attitude_star_list, attitude_star_list)

            alpha_numerator = np.exp(
                -np.multiply(self.confirmation_bias, np.abs(difference_matrix))
            )

            non_diagonal_weighting_matrix = (
                self.adjacency_matrix * alpha_numerator
            )  # We want only those values that have network connections

            norm_weighting_matrix = self.normlize_matrix(
                non_diagonal_weighting_matrix
            )  # normalize the matrix row wise

            weighting_matrix_list.append(norm_weighting_matrix)

        return weighting_matrix_list


    def calc_total_emissions_flow(self) -> int:
        """
        Calculate total carbon emissions of N*M behaviours

        Parameters
        ----------
        None

        Returns
        -------
        total_network_emissions: float
            total network emissions from each Individual object
        """
        total_network_emissions = sum(
            [x.individual_carbon_emissions_flow for x in self.agent_list]
        )
        return total_network_emissions

    def calc_network_identity(self) -> tuple[float, float, float, float]:
        """
        Return various identity properties, such as mean, variance, min and max

        Parameters
        ----------
        None

        Returns
        -------
        identity_list: list
            list of individuals identity 
        identity_mean: float
            mean of network identity at time step t
        identity_std: float
            std of network identity at time step t
        identity_variance: float
            variance of network identity at time step t
        identity_max: float
            max of network identity at time step t
        identity_min: float
            min of network identity at time step t
        """
        identity_list = [x.identity for x in self.agent_list]
        identity_mean = np.mean(identity_list)
        identity_std = np.std(identity_list)
        identity_variance = np.var(identity_list)
        identity_max = max(identity_list)
        identity_min = min(identity_list)
        return (identity_list,identity_mean, identity_std, identity_variance, identity_max, identity_min)

    def update_individuals(self):
        """
        Update Individual objects with new information regarding social interactions

        Parameters
        ----------
        None

        Returns
        -------
        None
        """
        pU,pC,pR = self.TA_component_matrix
        for i in range(self.N):
            self.agent_list[i].next_step(
                self.t, self.social_component_matrix[i], (pU[i], pC[i], pR[i])
            )

    def save_timeseries_data_network(self):
        """
        Save time series data

        Parameters
        ----------
        None

        Returns
        -------
        None
        """
        self.history_time.append(self.t)
        self.history_weighting_matrix.append(self.weighting_matrix)
        self.history_social_component_matrix.append(self.social_component_matrix)
        self.history_weighting_matrix_convergence.append(
            self.weighting_matrix_convergence
        )
        self.history_average_identity.append(self.average_identity)
        self.history_std_identity.append(self.std_identity)
        self.history_var_identity.append(self.var_identity)
        self.history_min_identity.append(self.min_identity)
        self.history_max_identity.append(self.max_identity)
        self.history_av_behaviour.append(self.av_behaviour)
        self.history_av_thresholds.append(self.av_thresholds)
        self.history_total_carbon_emissions_flow.append(self.total_carbon_emissions_flow)
        self.history_total_carbon_emissions_stock.append(self.total_carbon_emissions_stock)
        
        if self.alpha_change == ("static_culturally_determined_weights" or "dynamic_culturally_determined_weights"):
            self.history_total_identity_differences.append(self.total_identity_differences)

    def next_step(self):
        """
        Push the simulation forwards one time step. First advance time, then update individuals with data from previous timestep
        then produce new data and finally save it.

        Parameters
        ----------
        None

        Returns
        -------
        None
        """
        # advance a time step
        self.t += 1

        # execute step
        self.update_individuals()
        

        # update network parameters for next step
        if self.alpha_change == "dynamic_culturally_determined_weights":
            if self.save_timeseries_data:
                (
                    self.weighting_matrix,
                    self.total_identity_differences,
                    self.weighting_matrix_convergence,
                ) = self.update_weightings()
            else:
                self.weighting_matrix, self.total_identity_differences,__ = self.update_weightings()
        elif self.alpha_change == "behavioural_independence":#independent behaviours
            self.weighting_matrix_list = self.update_weightings_list()

        self.social_component_matrix = self.calc_social_component_matrix()
        self.TA_component_matrix = self.calc_TA_component_matrix()
        self.total_carbon_emissions_flow = self.calc_total_emissions_flow()
        self.total_carbon_emissions_stock += self.total_carbon_emissions_flow
        (
                self.identity_list,
                self.average_identity,
                self.std_identity,
                self.var_identity,
                self.min_identity,
                self.max_identity,
        ) = self.calc_network_identity()
        self.av_behaviour = np.mean([np.mean(a.attitudes) for a in self.agent_list])
        self.av_thresholds = np.mean([a.av_thresholds for a in self.agent_list])

        if (self.t % self.compression_factor == 0) and (self.save_timeseries_data):
            self.save_timeseries_data_network()
