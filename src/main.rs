extern crate rand;

use std::vec::Vec;

pub mod policy;
pub mod environment;
pub mod agent;
pub mod rl;
use rl::{QLearningActionSelector, SARSAActionSelector, EpsilonGreedy, SoftMaxExploration};

use policy::Policy;


fn main() {
    println!("Hello, world!");

    let env = environment::Env::new();
    let mut agent = agent::Agent::new(&env);
//    let policy = policy::RandomPolicy::new();
////    let policy = HumanControlPolicy::new();
////    println!("Evaluation of policy: {}", env.evaluate_policy(&*policy, 0.9, 0.001));
////    println!("Value iteration: {}", env.value_iteration(0.9, 0.001));
//    let (policy2, value) = env.policy_iteration(0.9, 0.001);
//    println!("Policy iteration: {}", value);
//    let result = policy2.solve(&env, & mut agent);
//
//    println!("Finished with result {}", result);

//    env.linear_programming(0.9);

//    let policy = rl::model_free_learning(&env, &mut QLearningActionSelector::new(EpsilonGreedy::new(0.05)), 0.01, 0.99, 0.0001);
//    let policy = rl::model_free_learning(&env, &mut QLearningActionSelector::new(SoftMaxExploration::new(20.0)), 0.01, 0.99, 0.0001);
//    let policy = rl::model_free_learning(&env, &mut SARSAActionSelector::new(EpsilonGreedy::new(0.05)), 0.01, 0.99, 0.0001);
    let policy = rl::model_free_learning(&env, &mut SARSAActionSelector::new(SoftMaxExploration::new(20.0)), 0.01, 0.99, 0.0001);
    policy.solve(&env, &mut agent);
}
