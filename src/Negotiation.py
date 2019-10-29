from NegotiationTools import NegTools, StatisticsLogger
from NegotiationConfig import *

class Agent():
    def __init__(self, agentname, model, confidence_func):
        '''
        :param agentname - Identifier of this agent
        :param model - Instance of SegmentationModel. Can be None if an initial proposal is provided.
        :param confidence_func - function that given a prediction [H, W, Label], returns the confidence in the initial proposal.
        '''
        
        self.agentname=agentname
        self.model=model
        self.task = None
        self.initial_proposal = None
        self.confidence_func = confidence_func
        self.alpha = None
        
    def new_task(self, input_image=None, initial_proposal=None):
        
        if input_image is not None and initial_proposal is None:
            self.task = image
            self.initial_proposal = self.model.predict(image).numpy()[0]
        elif input_image is None and initial_proposal is not None:
            self.initial_proposal = initial_proposal
        else:
            raise ValueError("Ambiguous input: Provide either an input image or an initial proposal")
        # Alpha is the "will to concede toward the agreement", so it's 1.0 - confidence in this case.
        self.alpha = 1.0 - self.confidence_func(self.initial_proposal)
        self.last_proposal = self.initial_proposal
        return self.initial_proposal
    
    def propose(self, agreement):
        self.last_agreement = agreement
        self.last_proposal = self.last_proposal + self.alpha*(agreement - self.last_proposal)
        return self.last_proposal
    
    
class Mediator():
    def __init__(self, agents):
        self.agents = agents
        self.last_step=0
        self.W = None
        self.tools = NegTools()
        
    def start_new_task(self, input_image=None, initial_proposals=None):
        '''
        Begins a new negotiation process. Called internally from the negotiation procedure.
        '''
        self.last_step=0
        if input_image is not None and initial_proposals is None:
            self.task = input_image
            self.initial_proposals = np.array([agent.new_task(input_image=self.task) for agent in self.agents])
        elif input_image is None and initial_proposals is not None:
            self.initial_proposals = np.array([agent.new_task(initial_proposal=agent_proposal) for agent, agent_proposal in zip(self.agents, initial_proposals)])
        else:
            raise ValueError("Ambiguous input: Provide either an input image or an initial prediction")       
        
        self.last_proposals = self.initial_proposals
        
        if self.W is None:
            self.W = np.ones_like(self.initial_proposals)
            
        return self.last_proposals

        
    def negotiation(self, input_image=None, initial_proposals=None, agent_weights=None, timeout = 10000):
        for i in range(self.last_step, self.last_step+timeout):
            if i==0:
                if agent_weights is not None:
                    self.W = agent_weights
                    
                if input_image is not None and initial_proposals is None:
                    self.last_proposals = self.start_new_task(input_image=input_image)
                elif input_image is None and initial_proposals is not None:
                    self.last_proposals = self.start_new_task(initial_proposals=initial_proposals)
                else:
                    raise ValueError("Ambiguous input: Provide either an input image or an initial prediction")               
            else:
                # Propose the new agreement to the agents
                
                self.last_proposals = np.array([agent.propose(self.last_agreement) for agent in self.agents]) # ((p0, u0), (p1, u1), ...)
            
            self.last_step = i            
            self.last_agreement = np.divide(np.sum(self.last_proposals*self.W, axis=0), np.sum(self.W, axis=0))

            if self.tools.get_consensus(self.last_proposals).all():
                yield 'consensus', self.last_agreement, self.last_proposals
                raise StopIteration()
            else:
                yield 'negotiation', self.last_agreement, self.last_proposals
        yield 'timeout', self.last_agreement, self.last_proposals
        raise StopIteration()
        

def run_negotiation_on_proposasls(sample_id, initial_proposals, ground_truth, confidence_functions, method_name, log_process, max_steps, agent_weights = None):    
    '''
    Run a negotiation starting from an initial proposals and a ground truth
    :param sample_id: an integer for identifying the current input sample. Will be reported in the log.
    :param initial_proposals: array of shape [Agents, H, W, Labels].
    :param ground_truth: Ground truth of shape [H, W, Labels] and type np.float. 
    :param confidence_functions: array of confidence functions that will be called by each agent. each function should take as input a proposal of shape [H,W,Labels] as input and return a confidence of shape [H, W].
    :param method_name: name of the negotiation method that is currently used
    :param log_process: [Bool] if True, the function logs statistics for the full process, otherwise it only returns the last agreement and proposals (determined from either the reach of consensus or the given timeout)
    :param max_steps: maximum steps of negotiations. If log_process is True and consensus is reached, the log will be padded up to max_steps for visualization consistency. If log_process is False, this can be set to a very high number and the negotiations stops only when consensus is reached.
    :param agent_weights: [Default:None] weights of the agents used in aggregation phase
    :return: If log_process is True: (Log DataFrame, Last Agreement, Last Proposals), otherwise (Last Agreement, Last Proposals)
    '''
    
    
    logger = StatisticsLogger()
    
    agents = [Agent(modelname, None, confidence_func) for modelname, confidence_func in zip(AGENT_NAMES, confidence_functions)]
    mediator = Mediator(agents)
    
    if log_process:
        next_step = enumerate(mediator.negotiation(initial_proposals=initial_proposals, agent_weights=agent_weights,timeout=max_steps))

        for step, (status, curr_agreement, curr_proposals) in next_step:

            logger.log_step(sample=sample_id, 
                            png_path='', 
                            seg_path='',
                            step=step,
                            method=method_name,
                            status=status,
                            agreement=curr_agreement,
                            proposals=curr_proposals,
                            ground_truth=ground_truth,
                            max_steps=max_steps,
                            binary_strategy='maximum'
                           )
            print("Sample:{} Step: {}".format(str(sample_id), str(step)))
        return logger.pd, curr_agreement, curr_proposals
    else:
        next_step = enumerate(mediator.negotiation(initial_proposals=initial_proposals, agent_weights=agent_weights,timeout=max_steps))

        for step, (status, curr_agreement, curr_proposals) in next_step:
            if status == 'consensus':
                break
            #print("Sample:{} Step: {}".format(str(sample_id), str(step)))
        return curr_agreement, curr_proposals