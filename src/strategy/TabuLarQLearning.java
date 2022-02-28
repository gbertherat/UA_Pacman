package strategy;

import java.util.ArrayList;
import java.util.List;
import java.util.Random;

import agent.AgentAction;
import motor.PacmanGame;

public class TabuLarQLearning extends QLearningStrategy {
	private final double epsilon, gamma, alpha;
	private final double[][] tab;

	public TabuLarQLearning(double epsilon, double gamma, double alpha, int slotsX, int slotsY) {
		super(epsilon, gamma, alpha);

		this.epsilon = epsilon;
		this.alpha = alpha;
		this.gamma = gamma;

		this.tab = new double[slotsX * slotsY][5];
		// TODO : Essayer de remplacer double[][] tab par une HashMap afin de prendre en compte l'ensemble des états possibles du jeu (Gum, Monstres, ..)
	}

	@Override
	public AgentAction chooseAction(PacmanGame state) {
		AgentAction action ;

		List<AgentAction> legalActions = new ArrayList<>();

		for (int i = 0; i < 5; i++) {
			if (state.isLegalMove(state.pacman, new AgentAction(i))) {
				legalActions.add(new AgentAction(i));
			}
		}

		if (Math.random() < epsilon) {
			action = legalActions.get(new Random().nextInt(legalActions.size()));
		} else {
			int index_x = state.pacman._position.getX();
			int index_y = state.pacman._position.getY();

			action = new AgentAction(getMaxFromTabForState((index_x * index_y - 1)));
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

		this.tab[(state.pacman._position.getX() * state.pacman._position.getY() - 1)][action.get_direction()] = reward + gamma * maxQNextState;
	}

	public int getMaxFromTabForState(int stateLine) {
		int max_index = 0;

		for (int i = 0; i < 5; ++i) {
			if (this.tab[stateLine][i] > max_index) {
				max_index = i;
			}
		}

		return max_index;
	}

	public double getMaxQ(PacmanGame state) {
		double max_value = 0;

		for (int i = 0; i < 5; ++i) {
			if (this.tab[(state.pacman._position.getX() * state.pacman._position.getY() - 1)][i] > max_value) {
				max_value = this.tab[(state.pacman._position.getX() * state.pacman._position.getY() - 1)][i];
			}
		}

		return max_value;
	}


}