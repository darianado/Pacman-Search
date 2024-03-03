# multiAgents.py
# --------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


from util import manhattanDistance
from game import Directions
import random, util

from game import Agent

class ReflexAgent(Agent):
    """
    A reflex agent chooses an action at each choice point by examining
    its alternatives via a state evaluation function.

    The code below is provided as a guide.  You are welcome to change
    it in any way you see fit, so long as you don't touch our method
    headers.
    """

    def getAction(self, gameState):
        """
        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {NORTH, SOUTH, WEST, EAST, STOP}
        """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices) # Pick randomly among the best
        
        # If 'Stop' is among the legal moves, consider removing it as a choice
        if len(legalMoves)>1 and legalMoves[chosenIndex]=='Stop':
            del legalMoves[chosenIndex]
            del scores[chosenIndex]
            # Recalculate the best score and indices without 'Stop'
            bestScore = max(scores)
            bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
            chosenIndex = random.choice(bestIndices) # Pick randomly among the best
        
        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState, action):
        """
        The evaluation function takes in the current and proposed successor
        GameStates (pacman.py) and returns a number, where higher numbers are better.
        """
        
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        
        newPos = successorGameState.getPacmanPosition()
        newFood = successorGameState.getFood()
        newGhostStates = successorGameState.getGhostStates()
        newGhostsPositions = successorGameState.getGhostPositions()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]
        newCapsulePosition = successorGameState.getCapsules()
        
        currGhostsPositions = currentGameState.getGhostPositions()
        currFood=currentGameState.getFood()
        
        score= successorGameState.getScore()
        
        newFoodDist = [util.manhattanDistance(newPos, food) for food in newFood.asList()]
        currFoodDist = [util.manhattanDistance(newPos, food) for food in currFood.asList()]
        
        newGhostDist = [util.manhattanDistance(newPos, g) for g in newGhostsPositions]
        currGhostDist = [util.manhattanDistance(newPos, g) for g in currGhostsPositions]
       
        # If this state leads to a win, return positive infinity
        if successorGameState.isWin():
            return float("inf")
        
        
        food_distance=0
        
        # Evaluate food distances
        for food in newFood.asList():
            distance = util.manhattanDistance(newPos, food)
            food_distance += (1/distance)*10 # Reward for being closer to food
        if len(newFood.asList())<len(currFood.asList()): 
             food_distance += 40# Additional reward if new food has been eaten
        if min(newFoodDist) < min(currFoodDist):
            food_distance += 20 # Reward if the nearest food is closer than before
            
        # Evaluate capsule distances
        for caps in newCapsulePosition:
            distance = util.manhattanDistance(newPos,caps)
            food_distance += (1/distance)*10
        if len(newCapsulePosition)<len(currentGameState.getCapsules()):#reward if pacman has eaten a capsule in this new state
             food_distance += 50
        
        
        ghost_distance=0
        
        # Evaluate ghost distances
        for i in range(len(newScaredTimes)):
            distance = util.manhattanDistance(newPos, newGhostsPositions[i])
            if newScaredTimes[i]>0: #Reward for eating scared ghost
                food_distance += (1/distance)*100
            else:
                if distance==0:# If Pacman touches an active ghost, return negative infinity
                    return float("-inf")
                else: #penalise ghost distances
                    if distance<3:
                        ghost_distance+=100/distance# Strong penalty for getting very close to an active ghost
                    else: ghost_distance += (1/distance)*10  # Penalty for getting closer to an active ghost
                    
        # Penalize for getting closer to an active ghost than in the previous state
        minNewGhostDist=min(newGhostDist)
        if newScaredTimes[newGhostDist.index(minNewGhostDist)]==0 and minNewGhostDist<min(currGhostDist):
            ghost_distance+=30      
                    
        # Combine scores to make the decision
        return score + food_distance- ghost_distance
                

def scoreEvaluationFunction(currentGameState):
    """
    This default evaluation function just returns the score of the state.
    The score is the same one displayed in the Pacman GUI.

    This evaluation function is meant for use with adversarial search agents
    (not reflex agents).
    """
    return currentGameState.getScore()

class MultiAgentSearchAgent(Agent):
    """
    This class provides some common elements to all of your
    multi-agent searchers.  Any methods defined here will be available
    to the MinimaxPacmanAgent, AlphaBetaPacmanAgent & ExpectimaxPacmanAgent.

    You *do not* need to make any changes here, but you can if you want to
    add functionality to all your adversarial search agents.  Please do not
    remove anything, however.

    Note: this is an abstract class: one that should not be instantiated.  It's
    only partially specified, and designed to be extended.  Agent (game.py)
    is another abstract class.
    """

    def __init__(self, evalFn = 'scoreEvaluationFunction', depth = '2'):
        self.index = 0 # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)


class MinimaxAgent(MultiAgentSearchAgent):
    """
    Your minimax agent (question 2)
    """
    def getAction(self, gameState):
        #we define the minimax function inside the getAction function. The minimax function is recursive, i.e., to return an answer,
        #it iterates over the children of each node of the tree applying the minimax function until reaching the terminal nodes of the branch.
        def minimax(gameState, depth, agentIndex):
            #the porgram stops if the depth is the maximum depth or if the gameState is a win or a lose state.
            if depth==self.depth or gameState.isWin() or gameState.isLose():
                return self.evaluationFunction(gameState)
            else:
            #if the agent is the pacman, i.e., index=0, the program wants to maximize the score. For that purpose, we need to iterate over all
            #its children, which is, possible actions to take, compute their score and then take the maximum over all the scores. 
            #Recursion makes sense here, because to know the scores of the children, we need to apply this exact same method with each one.
                if agentIndex==0:
                    maximize_list=[]
                    for action in gameState.getLegalActions(0):
                        #we save the value of each children which we compute through the minimax function
                        maximize_list.append(minimax(gameState.generateSuccessor(agentIndex, action),depth,agentIndex+1))
                    return max(maximize_list)
                #if the agent is the last ghost, we compute again the value of the children of that gameState and pick the minimum among them
                #in minimax we use depth+1 because when we iterate over pac man and all ghosts, we must pass to the next level in the tree.
                if agentIndex==gameState.getNumAgents() - 1:
                    minimize_list=[]
                    for action in gameState.getLegalActions(agentIndex):
                        minimize_list.append(minimax(gameState.generateSuccessor(agentIndex, action),depth+1,0))
                    return min(minimize_list)
                else:
                #if the agent is a ghost, but not the last one, for computing the value of the children of the current state
                #we perform minimax in each children and then pick the minimum score among the values of each children
                    minimize_list=[]
                    for action in gameState.getLegalActions(agentIndex):
                        minimize_list.append(minimax(gameState.generateSuccessor(agentIndex, action),depth,agentIndex+1))
                    return min(minimize_list)   
        #here there is the code that decides which actions the pacman needs to take        
        best_score = -1000
        best_action = None
        #iteration over all the actions the pacman can take given the current gameState 
        for action in gameState.getLegalActions(0):
            #save the score of the state generated after taking each action and choose the action with the highest score
            score = minimax(gameState.generateSuccessor(0, action),0,1)
            if score > best_score:
                best_score = score
                best_action = action

        return best_action

class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        
        #in this code we are doing the same as in minimax, with the only difference that for each minimax, we do not need to compute the 
        #value of all its children because of the alpha-beta variables which allow us to prune states which we know in advance 
        #that are not optimal.
        def minimax(gameState, depth, agentIndex,alpha,beta):
            if depth==self.depth or gameState.isWin() or gameState.isLose():
                return self.evaluationFunction(gameState)
            else:
                if agentIndex==0:
                    maximize_list=[]
                    for action in gameState.getLegalActions(0):
                        maximize_list.append(minimax(gameState.generateSuccessor(agentIndex, action),depth,agentIndex+1, alpha,beta))
                        #for each children of the current state, we define alpha in each iteration as the maximum of its previous value 
                        #and the maximum of the maximize list. If in some iteration alpha is greater than beta, then dont keep searching
                        #in the other children, because those states will never be better option for the pacman.
                        alpha=max(alpha, max(maximize_list))
                        if alpha>beta:
                            break
                    return max(maximize_list)
                if agentIndex==gameState.getNumAgents() - 1:
                    minimize_list=[]
                    for action in gameState.getLegalActions(agentIndex):
                        minimize_list.append(minimax(gameState.generateSuccessor(agentIndex, action),depth+1,0,alpha,beta))
                        #when the ghosts are playing we iterate over all the possible actions they can choose and in each iteration update 
                        #beta as the minimum between its previuos value and the minimum of the minimize list. If in some iteration beta 
                        #becomes smaller than alpha, we stop iterating on the other children (action over the current gameState)
                        beta=min(beta,min(minimize_list))
                        if alpha>beta:
                            break
                    return min(minimize_list)
                else:
                    minimize_list=[]
                    for action in gameState.getLegalActions(agentIndex):
                        minimize_list.append(minimax(gameState.generateSuccessor(agentIndex, action),depth,agentIndex+1,alpha,beta))
                        beta=min(beta,min(minimize_list))
                        if alpha>beta:
                            break
                    return min(minimize_list)   
        #in this part, there is an iteration over the actions the pacman can take in each state and chooses the one with the highest score
        # #taking into account that the pacman updates the alpha in each iteration as the maximum between its previous value and the maximum 
        # #of the maximize list.        
        best_score = -1000
        best_action = None
        alpha=-10000
        beta=10000
        for action in gameState.getLegalActions(0):
            score = minimax(gameState.generateSuccessor(0, action),0,1, alpha,beta)
            if score > best_score:
                best_score = score
                best_action = action
            alpha=max(alpha,best_score)

        return best_action


class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def getAction(self, gameState):
        """
        Returns the expectimax action using self.depth and self.evaluationFunction

        All ghosts should be modeled as choosing uniformly at random from their
        legal moves.
        """
        "*** YOUR CODE HERE ***"
        util.raiseNotDefined()

def betterEvaluationFunction(currentGameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: <write something here so we know what you did>
    """
    "*** YOUR CODE HERE ***"
    util.raiseNotDefined()

# Abbreviation
better = betterEvaluationFunction
