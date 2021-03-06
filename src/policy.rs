use std::io::{self, Read};
use std::io::prelude::*;
use ordered_float::NotNaN;
use ndarray::Array2;
use std::collections::HashMap;
use std::cmp::Ordering;

use crate::environment::{
    Movement,
    Pos,
    Env,
};
use crate::agent::Agent;

pub trait Policy
{
    fn new() -> Box<Self>;
    fn solve(&self, env: &Env, agent:&mut Agent) -> i32;
    fn prob(&self, env:&Env, pos: Pos, movement: &Movement) -> f32;
}

pub struct RandomPolicy {}

impl Policy for RandomPolicy
{
    fn new() -> Box<Self>
    {
//        let agent = Agent::new(&env);
        Box::new(Self { })
    }

    fn solve(&self, env: &Env, agent:&mut Agent) -> i32 {
        let mut s: i32= 0;
        while !env.is_terminal(agent.pos) {
            let movement = rand::random();
            s = agent.r#move(env, movement);
            print!("{:?} => {:?} {:?} \n", movement, s, agent.pos)
        }
        agent.cum_reward
    }

    fn prob(&self, env: &Env, pos: Pos, movement: &Movement) -> f32 {
        0.25
    }
}

struct HumanControlPolicy {
}

impl Policy for HumanControlPolicy
{
    fn new() -> Box<Self>
    {
        Box::new(Self {  })
    }

    fn solve(&self, env: &Env, agent:&mut Agent) -> i32 {
        let stdin = io::stdin();
        for line in stdin.lock().lines() {
            let line = line.unwrap();
            let movement = match line.as_str() {
                "w" => Some(Movement::Up),
                "s" => Some(Movement::Down),
                "a" => Some(Movement::Left),
                "d" => Some(Movement::Right),
                _ => None,
            };
            if let Some(movement) = movement {
                let s = agent.r#move(env, movement);
                print!("{:?} => {:?} {:?} \n", movement, s, agent.pos);

                if env.is_terminal(agent.pos) {
                    return agent.cum_reward;
                }
            }
        }

        panic!("Input finished before the agent could enter a final state");
    }

    fn prob(&self, env: &Env, pos: Pos, movement: &Movement) -> f32 {
        unimplemented!()
    }
}

struct TablePolicy {
    policy: HashMap<(Pos, Movement), f32>,
}

impl TablePolicy{

    // Initializes the table to assign uniform probability to all actions
    fn initialize(& mut self, env: &Env) {
        let actions = [Movement::Up, Movement::Down, Movement::Right, Movement::Left];
        for pos in env.iter_all_coordinates() {
            for a in actions.iter() {
                self.policy.insert((pos, *a), 0.25);
            }
        }
    }
}

impl Policy for TablePolicy{
    fn new() -> Box<Self> {
        Box::new(Self {
            policy: HashMap::new(),
        })
    }

    fn solve(&self, env: &Env, agent: &mut Agent) -> i32 {
        let actions = [Movement::Up, Movement::Down, Movement::Right, Movement::Left];
        while !env.is_terminal(agent.pos) {
            let r: f32 = rand::random();
            let mut tot_p = 0.0;
            for a in actions.iter() {
                tot_p += self.policy[&(agent.pos, *a)];
                if tot_p > r {
                    let s = agent.r#move(env, *a);
                    print!("{:?} => {:?} {:?} \n", a, s, agent.pos);
                    continue;
                }
            }
        }
        agent.cum_reward
    }

    fn prob(&self, env: &Env, pos: Pos, movement: &Movement) -> f32 {
        self.policy[&(pos, *movement)]
    }
}

// Represents deterministic policy
pub struct DetPolicy {
    pub policy: HashMap<Pos, Movement>,
}

impl DetPolicy{

    // Initializes the deterministic policy to always go up
    pub fn initialize(& mut self, env: &Env) {
        for pos in env.iter_all_coordinates() {
            self.policy.insert(pos, Movement::Up);
        }
    }

    pub fn print(&self, env: &Env) {
        use crate::environment::Cell;
        let size = env.size();
        print!("+");
        for y in 0..size.y { print!("--") }
        println!("-+");
        for x in 0..size.x {
            print!("|");
            for y in 0..size.y {
                let pos = Pos {x,y};
                let cell = env.cell(&pos);
                let text = match cell {
                    Cell::Final(r) => if *r > 0 { 'x' } else { ' ' },
                    Cell::Ice(_) => match self.policy.get(&pos).unwrap() {
                        Movement::Up => '???',
                        Movement::Down => '???',
                        Movement::Left => '???',
                        Movement::Right => '???',
                    },
                };
                print!(" {}", text);
            }
            println!(" |");
        }
        print!("+");
        for y in 0..size.y { print!("--") }
        println!("-+");
    }
}

impl Policy for DetPolicy{
    fn new() -> Box<Self> {
        Box::new(Self {
            policy: HashMap::new(),
        })
    }

    fn solve(&self, env: &Env, agent: &mut Agent) -> i32 {
        while !env.is_terminal(agent.pos) {
            let a = self.policy[&agent.pos];
            let s = agent.r#move(env, a);
            print!("{:?} => {:?} {:?} \n", a, s, agent.pos)
        }
        agent.cum_reward
    }

    fn prob(&self, env: &Env, pos: Pos, movement: &Movement) -> f32 {
        if self.policy[&pos] == *movement {1.0} else {0.0}
    }
}