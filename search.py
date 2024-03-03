# search.py
# ---------
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


"""
In search.py, you will implement generic search algorithms which are called by
Pacman agents (in searchAgents.py).
"""

import util

class SearchProblem:
    """
    This class outlines the structure of a search problem, but doesn't implement
    any of the methods (in object-oriented terminology: an abstract class).

    You do not need to change anything in this class, ever.
    """

    def getStartState(self):
        """
        Returns the start state for the search problem.
        """
        util.raiseNotDefined()

    def isGoalState(self, state):
        """
          state: Search state

        Returns True if and only if the state is a valid goal state.
        """
        util.raiseNotDefined()

    def getSuccessors(self, state):
        """
          state: Search state

        For a given state, this should return a list of triples, (successor,
        action, stepCost), where 'successor' is a successor to the current
        state, 'action' is the action required to get there, and 'stepCost' is
        the incremental cost of expanding to that successor.
        """
        util.raiseNotDefined()

    def getCostOfActions(self, actions):
        """
         actions: A list of actions to take

        This method returns the total cost of a particular sequence of actions.
        The sequence must be composed of legal moves.
        """
        util.raiseNotDefined()


def tinyMazeSearch(problem):
    """
    Returns a sequence of moves that solves tinyMaze.  For any other maze, the
    sequence of moves will be incorrect, so only use this for tinyMaze.
    """
    from game import Directions
    s = Directions.SOUTH
    w = Directions.WEST
    return  [s, s, w, s, w, w, s, w]

def depthFirstSearch(problem):
    """
    Search the deepest nodes in the search tree first.
    """
    
    # Create a stack data structure to use as the frontier for DFS
    frontier = util.Stack()
    
    # Get the start state of the problem
    start_state = problem.getStartState()
    
    # Initialize the frontier with the start state along with an empty path
    frontier.push((start_state, []))  # Each element in the frontier is a tuple (state, path to state)
    
    # Create a set to keep track of visited states for faster membership testing
    visited = set()  # Use a set for faster membership testing

    # Continue searching while the frontier is not empty
    while not frontier.isEmpty():
        state, current_path = frontier.pop()
        
        # If the current state is the goal state, return the path to reach it
        if problem.isGoalState(state):
            return current_path

        # Mark the current state as visited
        if state not in visited:
            visited.add(state)
            # Explore the successor states and actions
            for next_state, action, _ in problem.getSuccessors(state):
                # Check if the next state is not in the frontier or visited before
                if next_state not in frontier.list and next_state not in visited:
                    frontier.push((next_state, current_path + [action]))
    
    return [] 
    
 
def breadthFirstSearch(problem):
    """Search the shallowest nodes in the search tree first."""
    
    # Create a queue data structure to use as the frontier for BFS
    frontier = util.Queue()
    # Get the start state of the problem
    start_state = problem.getStartState()
    
    # Initialize the frontier with the start state along with an empty path
    frontier.push((start_state, []))  # Each element in the frontier is a tuple (state, path to state)
    
    # Create a set to keep track of visited states for faster membership testing
    visited = set()  # Use a set for faster membership testing

    # Continue searching while the frontier is not empty
    while not frontier.isEmpty():
        state, current_path = frontier.pop()
        
        # If the current state is the goal state, return the path to reach it
        if problem.isGoalState(state):
            return current_path

        # Mark the current state as visited
        if state not in visited:
            visited.add(state)
            # Explore the successor states and actions
            for next_state, action, _ in problem.getSuccessors(state):
                # Check if the next state is not in the frontier or visited before
                if next_state not in frontier.list and next_state not in visited:
                    frontier.push((next_state, current_path + [action]))
    
    return []  # If no path to the goal is found, return an empty list

def uniformCostSearch(problem):
    """Search the node of least total cost first."""
    
    #create the frontier as a PriorityQueue
    frontier = util.PriorityQueue()
    #introduce in the frontier the first state, i.e., first node
    frontier.push(problem.getStartState(),0)
    #create the visited set
    visited=set()
    #create a dictionary in which we the expanded nodes and the paths leading to them will appear
    paths={}
    paths[problem.getStartState()]=[]
    curr=problem.getStartState()
    
    # Continue searching while the frontier is not empty
    while not frontier.isEmpty():
        
        #we pick a state from the frontier
        curr= frontier.pop()
        
        # If the current state is the goal state, return the path to reach it
        if problem.isGoalState(curr):
            return paths[curr]
        
        #add this state to the visited set
        visited.add(curr)
        path=paths[curr]
        #compute the cost of reaching this node from the first node following
        #its actual path
        cost=problem.getCostOfActions(path)
        #iteration over the successors of current game state
        for succ, actions, new_cost in problem.getSuccessors(curr):
            #we store the cost of reaching each successor of curr through the path of curr
            priority=cost+new_cost
            #if the successor has been visited we do nothing
            if succ in visited:
                continue
            #if successor had already been visited we check if the new path is cheaper
            if succ in paths:
                if problem.getCostOfActions(paths[succ])>priority: 
                    frontier.update(succ, priority)
                    paths[succ] = paths[curr] + [actions]
            else:
                frontier.push(succ, priority)
                paths[succ]=paths[curr] + [actions]
          
    return paths[curr]

def nullHeuristic(state, problem=None):
    """
    A heuristic function estimates the cost from the current state to the nearest
    goal in the provided SearchProblem.  This heuristic is trivial.
    """
    return 0

def aStarSearch(problem, heuristic=nullHeuristic):
    """Search the node that has the lowest combined cost and heuristic first."""
    
    #this code is basically the same as uniform cost, taking into account the heuristic term in the cost
    
    frontier = util.PriorityQueue()
    frontier.push(problem.getStartState(),heuristic(problem.getStartState(), problem))
    visited=set()
    paths={}
    paths[problem.getStartState()]=[]
    curr=problem.getStartState()
    
    while not frontier.isEmpty():
        curr= frontier.pop()
        
        if problem.isGoalState(curr):
            return paths[curr]
        
        visited.add(curr)
        path=paths[curr]
        cost=problem.getCostOfActions(path)
        for succ, actions, new_cost in problem.getSuccessors(curr):
            priority=cost+new_cost+heuristic(succ,problem)#the priority function is now the total cost plus the heuristic term of the state
            if succ in visited:
                continue
            if succ in paths:
                #if the path to reach succ is cheaper passing through curr, update it
                if problem.getCostOfActions(paths[succ])+heuristic(succ,problem)>priority: 
                    frontier.update(succ, priority)
                    paths[succ] = paths[curr] + [actions]
            else:
                #if succ is not in paths, add it to the frontier with its path passing through curr.
                frontier.push(succ, priority)
                paths[succ]=paths[curr] + [actions]
          
    return paths[curr]


# Abbreviations
bfs = breadthFirstSearch
dfs = depthFirstSearch
astar = aStarSearch
ucs = uniformCostSearch
