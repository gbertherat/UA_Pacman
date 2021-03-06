package strategy;

import agent.AgentAction;
import motor.PacmanGame;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Random;


public class ApproximateQLearningStrategy extends QLearningStrategy{
	private int nFeatures;
	private int nActions;
	private double[] weights;
	private final double epsilon, gamma, alpha;

	public ApproximateQLearningStrategy(double epsilon, double gamma, double alpha) {
		super(epsilon, gamma, alpha);

		this.epsilon = epsilon;
		this.gamma = gamma;
		this.alpha = alpha;

		this.nFeatures = 4;
		this.nActions = 4;

		this.weights = new double[nFeatures+1];
		for(int i = 0; i < weights.length; i++){
			this.weights[i] = new Random().nextFloat();
		}
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

			for(int i = 0; i < nActions; i++){
				AgentAction curAction = new AgentAction(i);
				if(state.isLegalMove(state.pacman, curAction)) {
					double[] features = extractFeatures(state, curAction);
					double qValue = scalarProduct(weights, features);

					if (qValue > maxQ) {
						maxQ = qValue;
						action = curAction;
					} else if(qValue == maxQ){
						if(new Random().nextBoolean()){
							action = curAction;
						}
					}
				}
			}
		}

		return action;
	}

	@Override
	public void update(PacmanGame state, PacmanGame nextState, AgentAction action, double reward,
			boolean isFinalState) {

		double maxQnextState = -9999;
		double targetQ = reward;
		if (!isFinalState) {
			for (int i = 0; i < this.nActions; i++) {
				AgentAction a = new AgentAction(i);
				if (nextState.isLegalMove(nextState.pacman, a)) {
					double[] features = extractFeatures(nextState, a);
					double nextStateQ = scalarProduct(weights, features);

					if (nextStateQ > maxQnextState) {
						maxQnextState = nextStateQ;
					}
				}
			}

			targetQ = reward + gamma * maxQnextState;
		}
		double[] features = extractFeatures(state, action);
		double qValue = scalarProduct(weights, features);

		for (int i = 0; i < this.weights.length; i++) {
			this.weights[i] = this.weights[i] - 2 * this.alpha * features[i] * (qValue - targetQ);
		}

//		System.out.println(Arrays.toString(weights));
//		System.out.println(Arrays.toString(features));
//		System.out.println("---------");
	}

	private double scalarProduct(double[] weights, double[] features){
		double res = 0;
		for(int i = 0; i < weights.length; i++){
			res += weights[i] * features[i];
		}

		return res;
	}

	private double[] extractFeatures(PacmanGame state, AgentAction action){
		double[] features = new double[this.nFeatures+1];

		int nextX = state.pacman.get_position().getX() + action.get_vx();
		int nextY = state.pacman.get_position().getY() + action.get_vy();
		int x;

		switch (nFeatures + 1) {
			case 5:
				// D??tection de capsules proche du Pacman
				x = -1;
				while (x <= 1 && features[1] == 0) {
					if ((nextX + x > 0 && state.getMaze().getSizeX() > nextX + x && state.getMaze().isCapsule(nextX + x, nextY) )
							|| (nextY + x > 0 && state.getMaze().getSizeY() > nextY + x && state.getMaze().isCapsule(nextX, nextY + x))) {
						features[4] = 1;
					}
					x++;
				}
			case 4:
				if(state.isGhostsScarred()) {
					x = -1;
					while (x <= 1) {
						int i = x;
						if (state.getPostionFantom().stream().anyMatch(e -> e.getX() == nextX + i && e.getY() == nextY)
								|| state.getPostionFantom().stream().anyMatch(e -> e.getX() == nextX && e.getY() == nextY + i)) {
							features[3]++;
						}
						x++;
					}
				}
			case 3:
				// D??tection de fant??mes proche du Pacman
				if(!state.isGhostsScarred()) {
					x = -1;
					while (x <= 1) {
						int i = x;
						if (state.getPostionFantom().stream().anyMatch(e -> e.getX() == nextX + i && e.getY() == nextY)
								|| state.getPostionFantom().stream().anyMatch(e -> e.getX() == nextX && e.getY() == nextY + i)) {
							features[2]++;
						}
						x++;
					}
				}
			case 2:
				// D??tection de food proche du Pacman
				x = -1;
				while (x <= 1 && features[1] == 0) {
					if ((nextX + x > 0 && state.getMaze().getSizeX() > nextX + x && state.getMaze().isFood(nextX + x, nextY) )
							|| (nextY + x > 0 && state.getMaze().getSizeY() > nextY + x && state.getMaze().isFood(nextX, nextY + x))) {
						features[1] = 1;
					}
					x++;
				}
			case 1:
				// La feature 0 est toujours ?? 1
				features[0] = 1;

		}

//		System.out.println(Arrays.toString(features));
		return features;
	}
}