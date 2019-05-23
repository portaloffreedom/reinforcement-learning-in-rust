use rand::prelude::*;
use std::collections::hash_map::HashMap;
use crate::environment::{Movement, Pos, Env};
use crate::agent::Agent;
use crate::policy::{Policy, DetPolicy};
use std::fs::File;
use csv::Writer;
use std::collections::linked_list::LinkedList;

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

pub struct SoftMaxExploration {
    temperature: f32,
    rng: ThreadRng,
}

impl SoftMaxExploration {
    pub fn new(temperature: f32) -> Self {
        SoftMaxExploration {
            temperature,
            rng: thread_rng()
        }
    }
}

impl ExplorationStrategy for SoftMaxExploration {
    fn next_state(&mut self, state: Pos, state_value: &HashMap<(Pos, Movement), f32>) -> Movement {
        let actions = Movement::actions();
        let logits: Vec<f32> = actions.iter().map(
            |a| (state_value[&(state, *a)] / self.temperature).exp()).collect();
        let z: f32 = logits.iter().sum();
        let probs: Vec<f32> = logits.into_iter().map(|l| l/z).collect();
        let p: f32 = self.rng.gen();
        let mut p_sum = 0.0;
        let mut i = 0;
//        println!("{}, {}", p_sum, p);
        // Why the fuck is it possible for rng() to generate 0.0?
        while p >= p_sum && i < 4{
            i += 1;
            p_sum += probs[i-1];
        }
        actions[i-1]
    }
}

struct NoExplorationStrategy {}

impl ExplorationStrategy for NoExplorationStrategy {
    fn next_state(&mut self, state: Pos, state_value: &HashMap<(Pos, Movement), f32>) -> Movement {
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
pub struct SARSAActionSelector<T: ExplorationStrategy> {
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

struct StepSample {
    k: i32,
    l: usize,
    t: i32,
    position_start: Pos,
    action: Movement,
    reward: f32,
    position_end: Pos,
}

impl StepSample {
    pub fn new(k: i32, l: usize, t: i32, position_start: Pos, action: Movement, reward: f32, position_end: Pos) -> Self {
        Self {
            k,l,t,
            position_start,
            action,
            reward,
            position_end,
        }
    }
}

struct Memory {
    pub trajectory_length: usize,
    pub l: usize,
    pub k: i32,
    samples: LinkedList<StepSample>,
    pub memory_size: usize,
}

impl Memory {
    pub fn new(memory_size: usize, trajectory_length: usize) -> Self {
        Self {
            trajectory_length,
            l: 1,
            k: 0,
            samples: LinkedList::new(),
            memory_size,
        }
    }

    pub fn insert(&mut self, position_start: Pos, action: Movement, reward: f32, position_end: Pos) -> bool {
        let t = self.k - ((self.l - 1) * self.trajectory_length) as i32;

        //if too big, remove old stuff
        if self.samples.len() == self.memory_size {
            self.samples.pop_back();
        }

        self.samples.push_front(StepSample::new(
            self.k,
            self.l,
            t,
            position_start,
            action,
            reward,
            position_end)
        );

        self.k +=1;

        if self.k as usize == self.l*self.trajectory_length {
            self.l += 1;
            true
        } else {
            false
        }
    }

    pub fn random_sample(&self, rng: &mut rand::rngs::ThreadRng) -> &StepSample {
        self.samples
            .iter()
            .choose(rng)
            .expect("Cannot pick random sample from memory")
    }
}

pub fn model_free_learning (env: &Env, action_selector: &mut ActionSelector, step_size: f32, discount: f32, amt_episodes: i32, use_memory: Option<(usize, usize)>)
                            -> (DetPolicy, Vec<String>, Vec<String>)
{
    let mut rng = rand::thread_rng();
    let mut state_value: HashMap<(Pos, Movement), f32> = HashMap::new();
    let actions = Movement::actions();
    // Initialize Q-map
    for pos in env.iter_all_coordinates() {
        for action in actions.iter() {
            state_value.insert((pos, *action), 0.0);
        }
    }
    // Initialize memory
    let mut memory: Option<Memory> = use_memory.map(|(memory_size, trajectory_length)| Memory::new(memory_size, trajectory_length));

    let mut cum_reward = 0.0;
    let mut delta = 1000000.0;
//    let mut max_delta = 0.0001;

    let mut results_r = Vec::new();
    let mut results_e = Vec::new();

    // Iterate until convergence
//    let mut episode_num = 0;
//    while max_delta < delta {

    for episode_num in 0..amt_episodes {
        if episode_num % 500 == 0 {
//            println!("Episode {}", episode_num);
            results_r.push(cum_reward.to_string());
            results_e.push(env.evaluate_policy(
                    &policy_from_hashmap(&state_value, &env), discount, 0.001).to_string());
        }

        let mut agent = Agent::new(env);
        delta = 0.0;


        // Run a full episode, ie until the agent reaches a terminal state
        while !env.is_terminal(agent.pos) {
            let s = agent.pos;
            let a = action_selector.take_action(s, &state_value);
            let r = agent.r#move(env, a) as f32;
            cum_reward += r;
            let s_p = agent.pos;

//            println!("SARSA: ({:?}, {:?}, {:?}, {:?}, {:?})", s, a, r, s_p, a_p);

            if let Some(_memory) = &mut memory {
//                if _memory.insert(s, a, r, s_p) || env.is_terminal(s_p) {
//                  for i in 0..(_memory.memory_size*_memory.l*_memory.trajectory_length) {
                _memory.insert(s, a, r, s_p);

                let sample = _memory.random_sample( & mut rng);
                let future_action = action_selector.predict_action(sample.position_end, & state_value);
                delta = update_state_value_map( & mut state_value, discount, delta, step_size,
                sample.position_start,
                sample.action, sample.reward,
                sample.position_end,
                future_action);
            } else {
                let a_p = action_selector.predict_action(s_p, &state_value);
                delta = update_state_value_map(&mut state_value, discount, delta, step_size,s, a, r, s_p, a_p);
            }
        }
    }

    (policy_from_hashmap(&state_value, env), results_r, results_e)
}

pub fn double_q_learning (env: &Env, action_selector: &mut ActionSelector, step_size: f32, discount: f32, amt_episodes: i32)
                            -> (DetPolicy, Vec<String>, Vec<String>) {
    let mut state_value_a: HashMap<(Pos, Movement), f32> = HashMap::new();
    let mut state_value_b: HashMap<(Pos, Movement), f32> = HashMap::new();
    let mut avg_state_value : HashMap<(Pos, Movement), f32> = HashMap::new();
    let mut rng = thread_rng();

    let mut cum_reward = 0.0;
    let mut results_r = Vec::new();
    let mut results_e = Vec::new();

    let actions = Movement::actions();
    // Initialize Q-map
    for pos in env.iter_all_coordinates() {
        for action in actions.iter() {
            state_value_a.insert((pos, *action), 0.0);
            state_value_b.insert((pos, *action), 0.0);
            avg_state_value.insert((pos, *action), 0.0);
        }
    }
    // Iterate until convergence
    for episode_num in 0..amt_episodes {
        if episode_num % 500 == 0 {
            results_r.push(cum_reward.to_string());
            results_e.push(env.evaluate_policy(
                    &policy_from_hashmap(&avg_state_value, &env), discount, 0.01).to_string());
        }
//        println!("Episode {}", episode_num);

        let mut agent = Agent::new(env);

        // Run a full episode, ie until the agent reaches a terminal state
        while !env.is_terminal(agent.pos) {
            let s = agent.pos;
            // Use average state value over both estimates during action selection
            let a = action_selector.take_action(s, &avg_state_value);
            let r = agent.r#move(env, a) as f32;
            cum_reward += r;

            // Choose between Q^A and Q^B for the argmax in the next state and backup computation
            let mut state_value;
            let mut backup_q;
            if rng.gen::<f32>() > 0.5 {
                state_value = &mut state_value_a;
                backup_q = &mut state_value_b;
            }
            else {
                state_value = &mut state_value_b;
                backup_q = &mut state_value_a;
            }
            let s_p = agent.pos;
            let a_p = action_selector.predict_action(s_p, &state_value);

//            println!("SARSA: ({:?}, {:?}, {:?}, {:?}, {:?})", s, a, r, s_p, a_p);

            // Temporal difference using other Q value for backup
            let t_d = r + discount * backup_q[&(s_p, a_p)] - state_value[&(s, a)];
            state_value.insert((s, a),
                               // learning step
                               state_value[&(s, a)] + step_size * t_d);

            avg_state_value.insert((s, a), (state_value_a[&(s, a)] + state_value_b[&(s, a)]) / 2.0);
        }
    }

    (policy_from_hashmap(&avg_state_value, env), results_r, results_e)
}

fn update_state_value_map(state_value: &mut HashMap<(Pos, Movement), f32>, discount: f32, delta: f32, step_size: f32, s: Pos, a: Movement, r: f32, s_p: Pos, a_p: Movement)
                          -> f32
{
    let mut delta = delta;
    // Temporal difference
    let t_d = r + discount * state_value[&(s_p, a_p)] - state_value[&(s, a)];
    if t_d.abs() > delta {
        delta = t_d.abs()
    }
    state_value.insert((s, a),
                       // learning step
                       state_value[&(s, a)] + step_size * t_d
    );

    delta
}

fn policy_from_hashmap(state_value_map: &HashMap<(Pos, Movement), f32>, env: &Env) -> DetPolicy {
    let mut policy = DetPolicy::new();
    for state in  env.iter_all_coordinates() {
        let movement = Movement::actions()
            .into_iter()
            .fold(Movement::Right,|a, f| -> Movement {
                if state_value_map[&(state, f)] > state_value_map[&(state, a)] { f }
                else { a }
            });
        policy.policy.insert(state, movement);
    }

    *policy
}
