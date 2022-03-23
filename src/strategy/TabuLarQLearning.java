package strategy;

import java.util.*;
import java.util.stream.IntStream;

import agent.AgentAction;
import motor.PacmanGame;

public class TabuLarQLearning extends QLearningStrategy {
	private final double epsilon, gamma, alpha;
	private final HashMap<String, double[]> tab;
	private final int nActions;

	public TabuLarQLearning(double epsilon, double gamma, double alpha, int slotsX, int slotsY) {
		super(epsilon, gamma, alpha);

		this.epsilon = epsilon; // Random
		this.alpha = alpha; // Apprentissage
		// Alpha = 1 en cauchemar
		// 0.1 sinon
		this.gamma = gamma; // Récompense

		this.tab = new HashMap<>();
		this.nActions = 5;
	}

	@Override
	public AgentAction chooseAction(PacmanGame state) {
		AgentAction action ;

		List<AgentAction> legalActions = new ArrayList<>();

		for (int i = 0; i < this.nActions; i++) {
			if (state.isLegalMove(state.pacman, new AgentAction(i))) {
				legalActions.add(new AgentAction(i));
			}
		}

		if (Math.random() < epsilon) {
			action = legalActions.get(new Random().nextInt(legalActions.size()));
		} else {
			action = new AgentAction(getMaxFromTabForState(state));
		}

		return action;
	}

	@Override
	public void update(PacmanGame state, PacmanGame nextState, AgentAction action, double reward,
					   boolean isFinalState) {
		double maxQNextState;

		if (isFinalState) {
			maxQNextState = reward;
		}
		else {
			maxQNextState = getMaxQ(nextState);
		}

		String code = getMapCode(state);
		if(!tab.containsKey(code)){
			tab.put(code, new double[this.nActions]);
		}
		tab.get(code)[action.get_direction()] = (1-alpha) * this.tab.get(code)[action.get_direction()] + alpha * (reward + gamma * maxQNextState);
	}

	private String getMapCode(PacmanGame state){
		StringBuilder code = new StringBuilder();
		int cell = 0;
		for(int i = 0; i < state.getMaze().getSizeX(); i++){
			for(int j = 0; j < state.getMaze().getSizeY(); j++){
				if(state.getMaze().isWall(i, j)){
					continue;
				}

				int x = i;
				int y = j;

				int codeAction = 0;
				if(state.getMaze().isFood(x,y)) codeAction = 1;
				else if(state.getPostionFantom().stream().anyMatch(e -> e.getX() == x && e.getY() == y)) codeAction = 2;
				else if(state.getMaze().isCapsule(x,y)) codeAction = 3;
				else if(state.getPostionFantom().stream().anyMatch(e -> e.getX() == x && e.getY() == y)) codeAction = 4;

				code.append(codeAction);
				code.append(cell);
				cell++;
			}
		}
		code.append(state.isGhostsScarred() ? 1 : 0);

		return code.toString();
	}


	private double getMaxQ(PacmanGame state) {
		double max_value = 0;
		String code = getMapCode(state);

		for (int i = 0; i < this.nActions; i++) {
			if(tab.containsKey(code) && tab.get(code)[i] > max_value){
				max_value = tab.get(code)[i];
			}
		}

		return max_value;
	}

	public int getMaxFromTabForState(PacmanGame state) {
		int max_index = 0;
		String code = getMapCode(state);

		if(tab.containsKey(code)) {
			Integer[] indices = IntStream.range(0, tab.get(code).length).boxed().toArray(Integer[]::new);
			Arrays.sort(indices, Comparator.<Integer>comparingDouble(i -> tab.get(code)[i]).reversed());

			for(int indice : indices){
				if(state.isLegalMove(state.pacman, new AgentAction(indice))){
					return indice;
				}
			}
		}

		return max_index;
	}

}