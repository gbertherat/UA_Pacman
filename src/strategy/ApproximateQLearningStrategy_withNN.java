package strategy;

import agent.AgentAction;
import motor.PacmanGame;
import neuralNetwork.NeuralNetWorkDL4J;
import neuralNetwork.TrainExample;

import java.util.ArrayList;
import java.util.List;
import java.util.Random;


public class ApproximateQLearningStrategy_withNN extends NNQLearningStrats{
	private final int nFeatures;
	private final int nActions;
	private final double epsilon, gamma, alpha;
	private final int nEpochs;
	private final int batchSize;
	private final NeuralNetWorkDL4J nn;

	public ApproximateQLearningStrategy_withNN(double epsilon, double gamma, double alpha, int nEpochs, int batchSize) {
		super(epsilon, gamma, alpha);

		this.epsilon = epsilon;
		this.gamma = gamma;
		this.alpha = alpha;

		this.nFeatures = 4;
		this.nActions = 4;
		this.nn = new NeuralNetWorkDL4J(alpha, 0, this.nFeatures+1, 1);

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

			for(int i = 0; i < nActions; i++){
				AgentAction curAction = new AgentAction(i);
				if(state.isLegalMove(state.pacman, curAction)) {
					double[] features = extractFeatures(state, curAction);
					double qValue = this.nn.predict(features)[0];

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
		if (!isFinalState) {
			for (int i = 0; i < this.nActions; i++) {
				AgentAction a = new AgentAction(i);
				if (nextState.isLegalMove(nextState.pacman, a)) {
					double[] features = extractFeatures(nextState, a);
					double nextStateQ = this.nn.predict(features)[0];

					if (nextStateQ > maxQnextState) {
						maxQnextState = nextStateQ;
					}
				}
			}

			double[] targetQ = new double[1];
			targetQ[0] = reward + gamma * maxQnextState;

			double[] features = extractFeatures(state, action);
			TrainExample trainExample = new TrainExample(features, targetQ);
			trainExamples.add(trainExample);
		}
	}

	private double[] extractFeatures(PacmanGame state, AgentAction action){
		double[] features = new double[this.nFeatures+1];

		int nextX = state.pacman.get_position().getX() + action.get_vx();
		int nextY = state.pacman.get_position().getY() + action.get_vy();
		int x;

		switch (nFeatures + 1) {
			case 5:
				// Détection de capsules proche du Pacman
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
				// Détection de fantômes proche du Pacman
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
				// Détection de food proche du Pacman
				x = -1;
				while (x <= 1 && features[1] == 0) {
					if ((nextX + x > 0 && state.getMaze().getSizeX() > nextX + x && state.getMaze().isFood(nextX + x, nextY) )
							|| (nextY + x > 0 && state.getMaze().getSizeY() > nextY + x && state.getMaze().isFood(nextX, nextY + x))) {
						features[1] = 1;
					}
					x++;
				}
			case 1:
				// La feature 0 est toujours à 1
				features[0] = 1;

		}

//		System.out.println(Arrays.toString(features));
		return features;
	}

	@Override
	public void learn(ArrayList<TrainExample> trainExamples) {
		this.nn.fit(trainExamples, this.nEpochs, this.batchSize, this.alpha);
	}
}