package strategy;

import agent.AgentAction;
import motor.PacmanGame;


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
