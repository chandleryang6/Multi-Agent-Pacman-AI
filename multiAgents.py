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
from pacman import GameState

class ReflexAgent(Agent):
    """
    A reflex agent chooses an action at each choice point by examining
    its alternatives via a state evaluation function.

    The code below is provided as a guide.  You are welcome to change
    it in any way you see fit, so long as you don't touch our method
    headers.
    """


    def getAction(self, gameState: GameState):
        """
        You do not need to change this method, but you're welcome to.

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

        "Add more of your code here if you want to"

        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState: GameState, action):
        """
        Design a better evaluation function here.

        The evaluation function takes in the current and proposed successor
        GameStates (pacman.py) and returns a number, where higher numbers are better.

        The code below extracts some useful information from the state, like the
        remaining food (newFood) and Pacman position after moving (newPos).
        newScaredTimes holds the number of moves that each ghost will remain
        scared because of Pacman having eaten a power pellet.

        Print out these variables to see what you're getting, then combine them
        to create a masterful evaluation function.
        """
        # Useful information you can extract from a GameState (pacman.py)
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        newPos = successorGameState.getPacmanPosition()
        newFood = successorGameState.getFood()
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

        "*** YOUR CODE HERE ***"
        foodCoordinates = newFood.asList() #Creates list of the remaining food coordinates
        distancesToFood = [manhattanDistance(newPos, food) for food in foodCoordinates] #Creates list of the distances from Pacman's new position to each food coordinate
        foodScore = 1 / min(distancesToFood) if distancesToFood else 0 #Calculates the food score based on the closest food coordinate, set to 0 if there's no food left

        distancesToGhosts = [manhattanDistance(newPos, ghost.getPosition()) for ghost in newGhostStates] #Creates a list of the distances from Pacman's new position to each ghost position
        penaltyScore = 0 #Initializes penalty score
        for i, ghostDist in enumerate(distancesToGhosts): #For each of the ghost's distances 
            if newScaredTimes[i] == 0: #If ghost isn't scared
                if ghostDist > 0: #No negative distances
                    penaltyScore -= 1 / ghostDist #Calculates penalty score where the closer Pacman is to the ghost, the heavier the penalty 

        #Combines score from eating food/power pellets, score from Pacman getting close to food, and score from Pacman being close to non-scared ghost
        evaluationScore = successorGameState.getScore() + foodScore + penaltyScore 
        return evaluationScore



def scoreEvaluationFunction(currentGameState: GameState):
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

    def getAction(self, gameState: GameState):
        """
        Returns the minimax action from the current gameState using self.depth
        and self.evaluationFunction.

        Here are some method calls that might be useful when implementing minimax.

        gameState.getLegalActions(agentIndex):
        Returns a list of legal actions for an agent
        agentIndex=0 means Pacman, ghosts are >= 1

        gameState.generateSuccessor(agentIndex, action):
        Returns the successor game state after an agent takes an action

        gameState.getNumAgents():
        Returns the total number of agents in the game

        gameState.isWin():
        Returns whether or not the game state is a winning state

        gameState.isLose():
        Returns whether or not the game state is a losing state
        """
        "*** YOUR CODE HERE ***"
        def minimax(agentIndex, depth, gameState):
            if depth == self.depth or gameState.isWin() or gameState.isLose(): #Base case: if the max depth is reached or the game is over (Pacman wins/loses)
                return self.evaluationFunction(gameState) #Returns the game state 

            legalActions = gameState.getLegalActions(agentIndex) #Gets the legal moves for the current agent (either Pacman or the ghost)
            nextAgentIndex = (agentIndex + 1) % gameState.getNumAgents() #Calculates the next agent's turn
            nextDepth = depth + 1 if nextAgentIndex == 0 else depth #Increases depth after Pacman's turn

            if agentIndex == 0: #If the agent is Pacman (maximizing agent)
                return max(minimax(nextAgentIndex, nextDepth, gameState.generateSuccessor(agentIndex, action)) for action in legalActions) #Maximizes the minimax score for Pacman
            else: #If the agent is the ghost (minimizing agent)
                return min(minimax(nextAgentIndex, nextDepth, gameState.generateSuccessor(agentIndex, action)) for action in legalActions) #Minimizes the minimax score for the ghost

        bestAction = max(gameState.getLegalActions(0), key=lambda action: minimax(1, 0, gameState.generateSuccessor(0, action))) #Uses minimax to find the best action for Pacman (agent 0)
        return bestAction
        util.raiseNotDefined()



class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState: GameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        def alphaBeta(agentIndex, depth, gameState, alpha, beta):
            if depth == self.depth or gameState.isWin() or gameState.isLose(): #Base case: if the max depth is reached or the game is over (Pacman wins/loses)
                return self.evaluationFunction(gameState) #Returns the game state 

            legalActions = gameState.getLegalActions(agentIndex) #Gets the legal moves for the current agent (either Pacman or the ghost)
            nextAgentIndex = (agentIndex + 1) % gameState.getNumAgents() #Calculates the next agent's turn
            nextDepth = depth + 1 if nextAgentIndex == 0 else depth #Increases depth after Pacman's turn

            if agentIndex == 0: #If the agent is Pacman (maximizing agent)
                value = float('-inf') #Initializes 'value' to negative infinity
                for action in legalActions: #For all possible Pacman actions 
                    successorState = gameState.generateSuccessor(agentIndex, action) #Gets state after ghost takes action
                    value = max(value, alphaBeta(nextAgentIndex, nextDepth, successorState, alpha, beta)) #Recursively calls 'alphaBeta' to maximize the score
                    if value > beta: #If 'value' is > beta
                        return value #Prunes branch
                    alpha = max(alpha, value) #Updates alpha
                return value

            else: #If the agent is the ghost (minimizing agent)
                value = float('inf') #Initializes 'value' to infinity
                for action in legalActions: #For all possible ghost actions 
                    successorState = gameState.generateSuccessor(agentIndex, action) #Gets state after ghost takes action
                    value = min(value, alphaBeta(nextAgentIndex, nextDepth, successorState, alpha, beta)) #Recursively calls 'alphaBeta' to minimize the score
                    if value < alpha: #If 'value' is < alpha 
                        return value #Prunes branch
                    beta = min(beta, value) #Updates beta
                return value

        alpha = float('-inf') #Initializes alpha to negative infinity
        beta = float('inf') #Initializes beta to infinity
        bestAction = None  #Initializes the best action Pacman should take
        bestValue = float('-inf') #Initializes 'bestValue' to negative infinity (Pacman is maximizing)

        for action in gameState.getLegalActions(0): #For all legal actions Pacman (agent 0) can take 
            successorState = gameState.generateSuccessor(0, action) #Gets the state after Pacman takes action
            value = alphaBeta(1, 0, successorState, alpha, beta) #Calls 'alphaBeta' for the next agent at depth 0
            if value > bestValue: #If the value is better than the current best
                bestValue = value #Updates the best value
                bestAction = action #Uodates the best action 
            alpha = max(alpha, value) #Updates alpha for Pacman's moves

        return bestAction  
        util.raiseNotDefined()



class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def getAction(self, gameState: GameState):
        """
        Returns the expectimax action using self.depth and self.evaluationFunction

        All ghosts should be modeled as choosing uniformly at random from their
        legal moves.
        """
        "*** YOUR CODE HERE ***"
        def expectimax(agentIndex, depth, gameState):
            if depth == self.depth or gameState.isWin() or gameState.isLose(): #Base case: if the max depth is reached or the game is over (Pacman wins/lose
                return self.evaluationFunction(gameState) #Returns the game state 

            legalActions = gameState.getLegalActions(agentIndex) #Gets the legal moves for the current agent (either Pacman or the ghost)
            if not legalActions: #If no legal actions
                return self.evaluationFunction(gameState) #Returns the game state 

            nextAgentIndex = (agentIndex + 1) % gameState.getNumAgents() #Calculates the next agent's turn
            nextDepth = depth + 1 if nextAgentIndex == 0 else depth #Increases depth after Pacman's turn

            if agentIndex == 0: #If the agent is Pacman (maximizing agent)
                return max(expectimax(nextAgentIndex, nextDepth, gameState.generateSuccessor(agentIndex, action)) for action in legalActions) #Returns maximum of all successor states

            else: #If the agent is the ghost (minimizing agent)
                #returns the expected (average) value of all successor states
                return sum(expectimax(nextAgentIndex, nextDepth, gameState.generateSuccessor(agentIndex, action)) for action in legalActions) / len(legalActions) 

        #Uses expectimax to find the best action for Pacman (agent 0)
        bestAction = max(gameState.getLegalActions(0), key=lambda action: expectimax(1, 0, gameState.generateSuccessor(0, action))) 
        return bestAction
        util.raiseNotDefined()



def betterEvaluationFunction(currentGameState: GameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: <write something here so we know what you did>
    """
    "*** YOUR CODE HERE ***"
    currentPos = currentGameState.getPacmanPosition() #Gets Pacman's current postion
    currentFood = currentGameState.getFood() #Gets current food coordinates
    powerPellets = currentGameState.getCapsules() #Gets remaining power pellet coordinate
    ghostStates = currentGameState.getGhostStates() #Gets the state of all ghosts
    score = currentGameState.getScore() #Gets Pacman's current score

    foodCoordinates = currentFood.asList() #Creates list of the food coordinates
    if foodCoordinates: #If there are remaining food 
        distancesToFood = [manhattanDistance(currentPos, food) for food in foodCoordinates] #Creates list of the distances from Pacman's position to each food coordinate
        nearestFoodDist = min(distancesToFood) #Gets distance to the nearest food 
        score += 10.0 / nearestFoodDist #Adds a bonus to the score to encourage Pacman to go towards the nearest food

    for ghost in ghostStates: #For each ghost
        distanceToGhost = manhattanDistance(currentPos, ghost.getPosition()) #Gets distance from Pacman's position to ghost position
        if ghost.scaredTimer > 0: #If ghost is scared 
            score += 20.0 / distanceToGhost #Adds a bonus to the score to encourage Pacman to go eat the ghost
        else: #If Ghost is not scared
            if distanceToGhost <= 1: #If Pacman is close to ghost 
                score -= 1000.0 #Subtracts large penalty to the score where the closer Pacman is to the ghost, the heavier the penalty 
            else: #If Pacman is not that close to ghost
                score -= 5.0 / distanceToGhost #Subtracts small penalty to the score 

    distanceToPowerPellet = [manhattanDistance(currentPos, pellet) for pellet in powerPellets] # #Creates list of the distances from Pacman's position to each power pellet coordinate
    if distanceToPowerPellet: #If there are remaining power pellets
        nearestPowerPelletDist = min(distanceToPowerPellet) #Gets distance to the nearest power pellet 
        score += 15.0 / nearestPowerPelletDist #Adds a bonus to the score to encourage Pacman to go towards the nearest power pellet

    score -= 4.0 * len(foodCoordinates) #Subtracts penalty to the score for the remaining food
    score -= 10.0 * len(powerPellets) ##Subtracts penalty to the score for the remaining power pellets

    return score
    util.raiseNotDefined()

# Abbreviation
better = betterEvaluationFunction
