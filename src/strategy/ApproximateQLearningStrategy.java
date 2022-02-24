package strategy;

import java.util.ArrayList;
import java.util.List;

import agent.Agent;
import agent.AgentAction;
import agent.PositionAgent;
import motor.Maze;
import motor.PacmanGame;


import java.util.Random;


public class ApproximateQLearningStrategy extends QLearningStrategy{

	public ApproximateQLearningStrategy(double epsilon, double gamma, double alpha) {
		super(epsilon, gamma, alpha);
		// TODO Auto-generated constructor stub
	}

	@Override
	public AgentAction chooseAction(PacmanGame state) {
		// TODO Auto-generated method stub
		return null;
	}

	@Override
	public void update(PacmanGame state, PacmanGame nextState, AgentAction action, double reward,
			boolean isFinalState) {
		// TODO Auto-generated method stub
		
	}

	

	

}
