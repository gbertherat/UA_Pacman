package strategy;

import java.util.ArrayList;
import java.util.List;

import agent.Agent;
import agent.AgentAction;

import agent.PositionAgent;
import agent.typeAgent;
import motor.Maze;
import motor.PacmanGame;
import neuralNetwork.NeuralNetWorkDL4J;

import neuralNetwork.TrainExample;

import java.util.Random;


public class DeepQLearningStrategy extends NNQLearningStrats {
	private final double alpha, gamma, epsilon;

	private final int nActions;
	private int range;

	private final int nEpochs;
	private final int batchSize;
	private final NeuralNetWorkDL4J nn;

	public DeepQLearningStrategy(double epsilon, double gamma, double alpha, int range, int nEpochs, int batchSize) {
		super(epsilon, gamma, alpha);

		this.nActions = 4;
		this.range = range;
		this.nn = new NeuralNetWorkDL4J(alpha, 0, this.range*this.range, 2);

		this.epsilon = epsilon;
		this.gamma = gamma;
		this.alpha = alpha;

		this.nEpochs = nEpochs;
		this.batchSize = batchSize;
	}

	@Override
	public AgentAction chooseAction(PacmanGame state) {
		AgentAction action = new AgentAction(0);
		List<AgentAction> legalActions = new ArrayList<>();

		for (int i = 0; i < this.nActions; i++) {
			if (state.isLegalMove(state.pacman, new AgentAction(i))) {
				legalActions.add(new AgentAction(i));
			}
		}

		if (Math.random() < epsilon && this.isModeTrain()) {
			action = legalActions.get(new Random().nextInt(legalActions.size()));
		} else {
			double maxQ = -9999;
			int best_a = -1;
			double[] encodedState = getState(state);
			double[] output = this.nn.predict(encodedState);

			for(int i = 0; i < output.length; i++){
				if(output[i] > maxQ){
					maxQ = output[i];
					best_a = i;
				} else if(output[i] == maxQ){
					if(new Random().nextBoolean()){
						maxQ = output[i];
						best_a = i;
					}
				}
			}

			action = new AgentAction(best_a);
		}

		return action;
	}

	@Override
	public void update(PacmanGame state, PacmanGame nextState, AgentAction action, double reward,
					   boolean isFinalState) {

		double maxQnextState = -9999;
		if (!isFinalState) {
			for (int i = 0; i < this.nActions; i++) {
				AgentAction a = new AgentAction(i);
				if (nextState.isLegalMove(nextState.pacman, a)) {
					double[] encodedState = getState(nextState);
					double[] nextStateQ = this.nn.predict(encodedState);

					for (double v : nextStateQ) {
						if (v > maxQnextState) {
							maxQnextState = v;
						}
					}
				}
			}

			double[] encodedState = getState(state);
			double[] targetQ = this.nn.predict(encodedState);
			targetQ[action.get_direction()] = reward + gamma * maxQnextState;

			TrainExample trainExample = new TrainExample(encodedState, targetQ);
			trainExamples.add(trainExample);
		}
	}

	private double[] getState(PacmanGame game){
		double[] state = new double[this.range*this.range];

		int i = 0;
		for(int x = -this.range; x < this.range-1; x++){
			for(int y = -this.range; y < this.range-1; y++) {
				int pacmanX = game.pacman.get_position().getX();
				int pacmanY = game.pacman.get_position().getY();
				int relX = pacmanX + x;
				int relY = pacmanY + y;

				if(i >= 0 && i < range/4 && game.getMaze().isFood(relX, relY)){
					state[i] = 1;
				} else if(i >= range/4 && i < (2*range)/4 && game.getMaze().isCapsule(relX, relY)) {
					state[i] = 1;
				} else if(i >= (2*range)/4 && i < (3*range)/4 && game.getMaze().isWall(relX, relY)){
					state[i] = 1;
				} else if(i >= (3*range/4) && i < range && game.getPostionFantom().stream().anyMatch(e -> e.getX() == relX && e.getY() == relY)){
					state[i] = 1;
				} else {
					state[i] = 0;
				}

				i++;
			}
		}
		return state;
	}

	@Override
	public void learn(ArrayList<TrainExample> trainExamples) {
		// TODO Auto-generated method stub
		this.nn.fit(trainExamples, this.nEpochs, this.batchSize, this.alpha);
	}
	
	
}
