package strategy;

import java.util.*;

import agent.AgentAction;
import motor.PacmanGame;

public class TabuLarQLearning extends QLearningStrategy {
	private final double epsilon, gamma, alpha;
	private final HashMap<String, Double> tab;
	private final int nActions;

	public TabuLarQLearning(double epsilon, double gamma, double alpha, int slotsX, int slotsY) {
		super(epsilon, gamma, alpha);

		this.epsilon = epsilon; // Random
		this.alpha = alpha; // Apprentissage
		// Alpha = 1 en cauchemar
		// 0.1 sinon
		this.gamma = gamma; // Récompense

		this.tab = new HashMap<>();
		this.nActions = 4;
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

		String code = getMapCode(state, action.get_direction());
		tab.put(code, (1-alpha) * this.tab.getOrDefault(code, 0.0) + alpha * (reward + gamma * maxQNextState));
//		System.out.println(tab);
	}

	private String getMapCode(PacmanGame state, int direction){
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
		code.append(direction);

		return code.toString();
	}


	private double getMaxQ(PacmanGame state) {
		double max_value = 0;
		String code = getMapCode(state, 5);
		code = code.substring(0, code.length()-1);

		for (int i = 0; i < this.nActions; i++) {
			String tCode = code + i;
			if(tab.containsKey(tCode) && tab.get(tCode) > max_value){
				max_value = tab.get(tCode);
			}
		}

		return max_value;
	}

	public int getMaxFromTabForState(PacmanGame state) {
		int max_index = 5;

		String code = getMapCode(state, 5);
		code = code.substring(0, code.length()-1);

		for (int i = 0; i < this.nActions; i++) {
			if (state.isLegalMove(state.pacman, new AgentAction(i))) {
				String codeI = code + i;
				String codeMax = code + max_index;

				if (tab.containsKey(codeI) && tab.containsKey(codeMax) && tab.get(codeI) > tab.get(codeMax)) {
					max_index = i;
				}
			}
		}

		return max_index;
	}

}