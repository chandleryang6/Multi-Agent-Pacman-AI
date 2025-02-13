# **Multi-Agent Pacman AI**
This project implements **adversarial search algorithms** to control Pacman and ghost agents in a **multi-agent environment**. The project includes **Minimax, Alpha-Beta Pruning, and Expectimax** search strategies, as well as a custom evaluation function for better decision-making.

## **Project Overview**
The goal of this project is to build **AI agents** that can **intelligently navigate and compete in the Pacman environment**. The AI must decide the best moves by considering food, ghosts, and power pellets while predicting the behavior of adversarial agents (ghosts).

## **Key Features**
- **Reflex Agent:** A basic agent that reacts to its immediate surroundings.
- **Minimax Algorithm:** Implements a depth-based game tree to find the optimal move.
- **Alpha-Beta Pruning:** Optimized version of Minimax to reduce unnecessary computations.
- **Expectimax Algorithm:** Models stochastic ghost behavior to improve decision-making.
- **Custom Evaluation Function:** A scoring function that prioritizes survival, food collection, and ghost avoidance.

---

## **Implemented AI Agents**
### **1. Reflex Agent (`ReflexAgent`)**
A reactive agent that chooses actions based on an evaluation function that considers:
- Distance to the closest food
- Distance to ghosts (and whether they are scared)
- Number of remaining power pellets

```python
def evaluationFunction(self, currentGameState, action):
    successorGameState = currentGameState.generatePacmanSuccessor(action)
    newPos = successorGameState.getPacmanPosition()
    newFood = successorGameState.getFood().asList()
    
    foodScore = 1 / min([manhattanDistance(newPos, food) for food in newFood]) if newFood else 0

    ghostDistances = [manhattanDistance(newPos, ghost.getPosition()) for ghost in successorGameState.getGhostStates()]
    ghostPenalty = sum([-1 / dist for dist in ghostDistances if dist > 0])

    return successorGameState.getScore() + foodScore + ghostPenalty
```

### **2. Minimax Agent (`MinimaxAgent`)**
- Implements Minimax search where Pacman is the maximizing agent and ghosts are minimizing agents.
- Uses recursive depth-based search to compute the best move.

```python
def minimax(agentIndex, depth, gameState):
    if depth == self.depth or gameState.isWin() or gameState.isLose():
        return self.evaluationFunction(gameState)

    legalActions = gameState.getLegalActions(agentIndex)
    nextAgentIndex = (agentIndex + 1) % gameState.getNumAgents()
    nextDepth = depth + 1 if nextAgentIndex == 0 else depth

    if agentIndex == 0:  # Pacman (Maximizer)
        return max(minimax(nextAgentIndex, nextDepth, gameState.generateSuccessor(agentIndex, action)) for action in legalActions)
    else:  # Ghost (Minimizer)
        return min(minimax(nextAgentIndex, nextDepth, gameState.generateSuccessor(agentIndex, action)) for action in legalActions)
```

### **3. Alpha-Beta Pruning (`AlphaBetaAgent`)**
- Optimized Minimax algorithm using α (alpha) and β (beta) bounds to prune unnecessary branches.
- Improves efficiency while keeping optimal decision-making.

```python
def alphaBeta(agentIndex, depth, gameState, alpha, beta):
    if depth == self.depth or gameState.isWin() or gameState.isLose():
        return self.evaluationFunction(gameState)

    legalActions = gameState.getLegalActions(agentIndex)
    nextAgentIndex = (agentIndex + 1) % gameState.getNumAgents()
    nextDepth = depth + 1 if nextAgentIndex == 0 else depth

    if agentIndex == 0:  # Pacman (Maximizer)
        value = float('-inf')
        for action in legalActions:
            value = max(value, alphaBeta(nextAgentIndex, nextDepth, gameState.generateSuccessor(agentIndex, action), alpha, beta))
            if value > beta:
                return value  # Prune branch
            alpha = max(alpha, value)
        return value
    else:  # Ghost (Minimizer)
        value = float('inf')
        for action in legalActions:
            value = min(value, alphaBeta(nextAgentIndex, nextDepth, gameState.generateSuccessor(agentIndex, action), alpha, beta))
            if value < alpha:
                return value  # Prune branch
            beta = min(beta, value)
        return value
```

### **4. Expectimax Agent (`ExpectimaxAgent`)**
- Unlike Minimax, Expectimax does not assume ghosts play optimally.
- Instead, it models ghosts as stochastic agents, making random moves.

```python
def expectimax(agentIndex, depth, gameState):
    if depth == self.depth or gameState.isWin() or gameState.isLose():
        return self.evaluationFunction(gameState)

    legalActions = gameState.getLegalActions(agentIndex)
    nextAgentIndex = (agentIndex + 1) % gameState.getNumAgents()
    nextDepth = depth + 1 if nextAgentIndex == 0 else depth

    if agentIndex == 0:  # Pacman (Maximizer)
        return max(expectimax(nextAgentIndex, nextDepth, gameState.generateSuccessor(agentIndex, action)) for action in legalActions)
    else:  # Ghost (Chance Node)
        return sum(expectimax(nextAgentIndex, nextDepth, gameState.generateSuccessor(agentIndex, action)) for action in legalActions) / len(legalActions)
```

### **5. Custom Evaluation Function (`betterEvaluationFunction`)**
This function improves Pacman’s strategy by:
- Prioritizing food collection
- Avoiding dangerous ghosts
- Encouraging power pellet usage
- Balancing risk and reward

```python
def betterEvaluationFunction(currentGameState):
    currentPos = currentGameState.getPacmanPosition()
    currentFood = currentGameState.getFood().asList()
    powerPellets = currentGameState.getCapsules()
    ghostStates = currentGameState.getGhostStates()
    score = currentGameState.getScore()

    if currentFood:
        nearestFoodDist = min(manhattanDistance(currentPos, food) for food in currentFood)
        score += 10.0 / nearestFoodDist  # Encourages food collection

    for ghost in ghostStates:
        ghostDist = manhattanDistance(currentPos, ghost.getPosition())
        if ghost.scaredTimer > 0:
            score += 20.0 / ghostDist  # Encourages chasing scared ghosts
        else:
            score -= 1000.0 / ghostDist if ghostDist <= 1 else 5.0 / ghostDist  # Avoids active ghosts

    if powerPellets:
        nearestPelletDist = min(manhattanDistance(currentPos, pellet) for pellet in powerPellets)
        score += 15.0 / nearestPelletDist  # Encourages power pellet usage

    return score
```

## **How to Run the Project**
```python
# Play Pacman manually
python pacman.py

# Run Reflex Agent
python pacman.py -p ReflexAgent

# Run Minimax Agent
python pacman.py -p MinimaxAgent -a depth=3

# Run Alpha-Beta Agent
python pacman.py -p AlphaBetaAgent -a depth=3

# Run Expectimax Agent
python pacman.py -p ExpectimaxAgent -a depth=3

# Run Custom Evaluation Function
python pacman.py -p ExpectimaxAgent -a evalFn=betterEvaluationFunction
```

## **Technologies Used**
- Python
- AI Search Algorithms (Minimax, Alpha-Beta Pruning, Expectimax)
- Game AI for Pacman
