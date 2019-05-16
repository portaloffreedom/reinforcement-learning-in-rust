use rand::prelude::*;
use std::collections::hash_map::HashMap;
use crate::environment::{Movement, Pos, Env};
use crate::agent::Agent;

struct RandomExplorationStrategy {}
struct EpsilonGreedyExplorationStrategy {}
struct SoftMaxExplorationStrategy {}

pub trait ExplorationStrategy {
    fn next_state(&mut self, state: Pos, state_value: &HashMap<(Pos, Movement), f32>) -> Movement;
}

pub struct EpsilonGreedy {
    epsilon: f32,
    rng: ThreadRng,
}

impl EpsilonGreedy {
    pub fn new(epsilon: f32) -> Self {
        EpsilonGreedy{
            epsilon,
            rng: thread_rng()
        }
    }
}

impl ExplorationStrategy for EpsilonGreedy {
    fn next_state(&mut self, state: Pos, state_value: &HashMap<(Pos, Movement), f32>) -> Movement {
        if self.rng.gen::<f32>() < self.epsilon {
            return self.rng.gen::<Movement>();
        }
        max_value_next_action(state, state_value)
    }
}

fn max_value_next_action(state: Pos, state_value: &HashMap<(Pos, Movement), f32>) -> Movement {
    Movement::actions().into_iter().fold(Movement::Up,|a, f| -> Movement {
        if state_value[&(state, f)] > state_value[&(state, a)] { f }
        else { a }
    })
}
pub struct QLearningActionSelector<T: ExplorationStrategy> {
    exploration_strategy: T
}
struct SARSAActionSelector<T: ExplorationStrategy> {
    exploration_strategy: T
}

impl<T: ExplorationStrategy> QLearningActionSelector<T> {
    pub fn new(exploration_strategy: T) -> Self {
        Self {
            exploration_strategy
        }
    }
}

impl<T: ExplorationStrategy> SARSAActionSelector<T> {
    pub fn new(exploration_strategy: T) -> Self {
        Self {
            exploration_strategy
        }
    }
}

pub trait ActionSelector {
    fn take_action(&mut self, state: Pos, state_value: &HashMap<(Pos, Movement), f32>) -> Movement;
    fn predict_action(&mut self, state: Pos, state_value: &HashMap<(Pos, Movement), f32>) -> Movement;
}

impl<T: ExplorationStrategy> ActionSelector for QLearningActionSelector<T> {
    fn take_action(&mut self, state: Pos, state_value: &HashMap<(Pos, Movement), f32>) -> Movement {
        self.exploration_strategy.next_state(state, state_value)
    }

    fn predict_action(&mut self, state:Pos, state_value: &HashMap<(Pos, Movement), f32>) -> Movement {
        max_value_next_action(state, state_value)
    }
}

impl<T: ExplorationStrategy> ActionSelector for SARSAActionSelector<T> {
    fn take_action(&mut self, state: Pos, state_value: &HashMap<(Pos, Movement), f32>) -> Movement {
        self.exploration_strategy.next_state(state, state_value)
    }

    fn predict_action(&mut self, state:Pos, state_value: &HashMap<(Pos, Movement), f32>) -> Movement {
        self.exploration_strategy.next_state(state, state_value)
    }
}

pub fn model_free_learning (env: &Env, action_selector: &mut ActionSelector, step_size: f32, discount: f32, max_delta: f32) {
    let mut state_value: HashMap<(Pos, Movement), f32> = HashMap::new();
    let actions = Movement::actions();
    // Initialize Q-map
    for pos in env.iter_all_coordinates() {
        for action in actions.iter() {
            state_value.insert((pos, *action), 0.0);
        }
    }
    let mut delta = max_delta + 1.0;
    // Iterate until convergence
    while delta > max_delta {
        let mut agent = Agent::new(env);
        delta = 0.0;

        println!("New episode");

        // Run a full episode, ie until the agent reaches a terminal state
        while !env.is_terminal(agent.pos) {
            let s = agent.pos;
            let a = action_selector.take_action(s, &state_value);
            let r = agent.r#move(env, a) as f32;
            let s_p = agent.pos;
            let a_p = action_selector.predict_action(s_p, &state_value);

            println!("SARSA: ({:?}, {:?}, {:?}, {:?}, {:?})", s, a, r, s_p, a_p);

            // Temporal difference
            let t_d = r + discount * state_value[&(s_p, a_p)] - state_value[&(s, a)];
            if t_d.abs() > delta {
                delta = t_d.abs()
            }
            state_value.insert((s, a),
               // learning step
               state_value[&(s, a)] + step_size * t_d
            );
        }
    }
}

