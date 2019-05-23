extern crate rand;

use std::vec::Vec;

pub mod policy;
pub mod environment;
pub mod agent;
pub mod rl;
use rl::{QLearningActionSelector, SARSAActionSelector, EpsilonGreedy, SoftMaxExploration};

use policy::Policy;
use csv::Writer;
use std::fs::File;
use crate::policy::DetPolicy;


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

    let epsilon = 0.05;
    let discount = 0.99;
    let amt_episodes = 200000;
    let temperature = 20.0;
    let step_size = 0.01;

//    let policy = rl::model_free_learning(&env, &mut QLearningActionSelector::new(EpsilonGreedy::new(epsilon)), step_size, discount, amt_episodes);
//    let policy = rl::model_free_learning(&env, &mut QLearningActionSelector::new(SoftMaxExploration::new(temperature)), step_size, discount, amt_episodes);
//    let policy = rl::model_free_learning(&env, &mut SARSAActionSelector::new(EpsilonGreedy::new(epsilon)), step_size, discount, amt_episodes);
//    let policy = rl::model_free_learning(&env, &mut SARSAActionSelector::new(SoftMaxExploration::new(temperature)), step_size, discount, amt_episodes);
//    let policy = rl::double_q_learning(&env, &mut QLearningActionSelector::new(EpsilonGreedy::new(epsilon)), step_size, discount, amt_episodes);
//    policy.solve(&env, &mut agent);
//    policy.print(&env);

    write_statistics("q_learning_epsilon", &mut|| {
        rl::model_free_learning(&env, &mut QLearningActionSelector::new(EpsilonGreedy::new(epsilon)), step_size, discount, amt_episodes)
    });
    write_statistics("q_learning_softmax", &mut|| {
        rl::model_free_learning(&env, &mut QLearningActionSelector::new(SoftMaxExploration::new(temperature)), step_size, discount, amt_episodes)
    });
    write_statistics("SARSA_epsilon", &mut|| {
        rl::model_free_learning(&env, &mut SARSAActionSelector::new(EpsilonGreedy::new(epsilon)), step_size, discount, amt_episodes)
    });
    write_statistics("SARSA_softmax", &mut|| {
        rl::model_free_learning(&env, &mut SARSAActionSelector::new(SoftMaxExploration::new(temperature)), step_size, discount, amt_episodes)
    });
    write_statistics("double_q_epsilon", &mut|| {
        rl::double_q_learning(&env, &mut QLearningActionSelector::new(EpsilonGreedy::new(epsilon)), step_size, discount, amt_episodes)
    });
    write_statistics("double_q_softmax", &mut|| {
        rl::double_q_learning(&env, &mut QLearningActionSelector::new(SoftMaxExploration::new(temperature)), step_size, discount, amt_episodes)
    });
}

fn write_statistics(name: &str, closure: &mut Fn() -> (DetPolicy, Vec<String>, Vec<String>)) {
    let mut wrt_r = Writer::from_path(name.to_owned() + "_rewards.csv").expect("Failed to open file");
    let mut wrt_s = Writer::from_path(name.to_owned() + "_evals.csv").expect("Failed to open file");
    for i in 0..100 {
        let (policy, results_r, results_e) = closure();
        wrt_r.write_record(&results_r).expect("Failed to write records");
        wrt_s.write_record(&results_e).expect("Failed to write records");
    }
    wrt_r.flush().expect("Failed to write to file");
    wrt_s.flush().expect("Failed to write to file");
}
