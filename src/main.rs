extern crate rand;

use std::vec::Vec;

pub mod policy;
pub mod environment;
pub mod agent;

use policy::Policy;

fn main() {
    println!("Hello, world!");

    let env = environment::Env::new();
    let mut agent = agent::Agent::new(&env);
    let policy = policy::RandomPolicy::new();
//    let policy = HumanControlPolicy::new();
    println!("Evaluation of policy: {}", (&env).evaluate_policy(&*policy, 0.9, 0.001));
    println!("Value iteration: {}", (&env).value_iteration(0.9, 0.001));
    println!("Policy iteration: {}", (&env).policy_iteration(0.9, 0.001));
    let result = policy.solve(&env, & mut agent);

    println!("Finished with result {}", result);
}
