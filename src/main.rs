extern crate rand;

use std::io::{self, Read};
use std::io::prelude::*;
use std::vec::Vec;
use rand::{
    distributions::{Distribution, Standard},
    Rng,
};
use ordered_float::NotNaN;

use ndarray::Array2;
//use itertools::Itertools;
use std::collections::HashMap;

#[derive(Debug)]
pub enum Cell {
    Ice(i32),
    Final(i32),
}

impl Cell {
    pub fn reward(&self) -> i32
    {
        match self {
            Cell::Ice(reward) => reward.clone(),
            Cell::Final(reward) => reward.clone(),
        }
    }
}

// Action
#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash)]
pub enum Movement {
    Up,
    Right,
    Down,
    Left,
}

#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash)]
pub struct Pos {
    x: usize,
    y: usize,
}


impl Movement {
    pub fn into_vector(self) -> (isize, isize)
    {
        match self {
            Movement::Up    => (-1, 0),
            Movement::Down  => ( 1, 0),
            Movement::Left  => ( 0,-1),
            Movement::Right => ( 0, 1),
        }
    }
}

impl Distribution<Movement> for Standard {
    fn sample<R: Rng + ?Sized>(&self, rng: &mut R) -> Movement {
        match rng.gen_range(0, 4) {
            0 => Movement::Up,
            1 => Movement::Right,
            2 => Movement::Down,
            _ => Movement::Left,
        }
    }
}

pub struct Env {
    map: Vec<Vec<Cell>>,
    start: Pos,
    transition: HashMap<(Pos, Movement), Vec<(Pos, f32)>>,
}

use std::cmp::Ordering;


impl Env {
    pub fn new() -> Self
    {
        let mut env = Self {
            map: vec![
                vec![Cell::Ice(0), Cell::Ice(0),     Cell::Ice(0),     Cell::Final(100)],
                vec![Cell::Ice(0), Cell::Final(-10), Cell::Ice(0),     Cell::Final(-10)],
                vec![Cell::Ice(0), Cell::Ice(0),     Cell::Ice(20),    Cell::Final(-10)],
                vec![Cell::Ice(0), Cell::Final(-10), Cell::Final(-10), Cell::Final(-10)],
            ],
            start: Pos{x:3, y:0},
            transition: HashMap::new(),
        };
        (& mut env).setup_transition_map();
        env
    }

    fn setup_transition_map(& mut self) {
        self.transition = HashMap::new();
        let actions = [Movement::Up, Movement::Down, Movement::Right, Movement::Left];
        for pos in self.iter() {
            for movement in actions.iter() {
                let mut options: Vec<(Pos, f32)> = Vec::new();

                let movement_vec = movement.into_vector();
                let boundaries = self.size();
                let (new_pos, wall_hit) = self.check_movement(pos, movement_vec);

                let cell = &self.map[new_pos.x][new_pos.y];
                match cell {
                    Cell::Ice(_) => {
                        let mut wall_hit = false;
                        let mut other_pos = new_pos;
                        while !wall_hit {
                            let (_new_pos, _wall_hit) = self.check_movement(other_pos, movement_vec);
                            other_pos = _new_pos;
                            wall_hit = _wall_hit;
                            if let Cell::Ice(reward) = *(&self.map[other_pos.x][other_pos.y]) { }
                            else {
                                wall_hit = true;
                            }
                        }
                        if other_pos.x != new_pos.x || other_pos.y != new_pos.y {
                            options.push((new_pos, 0.95));
                            options.push((other_pos, 0.05))
                        } else {
                            options.push((new_pos, 1.0));
                        }
                    }
                    _ => {
                        options.push((new_pos, 1.0));
                    },
                }
                self.transition.insert((pos, movement.clone()), options);
            }
        }
        println!("Map setup");
    }

    fn reward(&self, pos: &Pos) -> i32 {
        self.map[pos.x][pos.y].reward()
    }

    pub fn size(&self) -> Pos
    {
        Pos{x:self.map.len(), y:self.map[0].len()}
    }

    pub fn start_pos(&self) -> Pos { self.start }

    fn check_movement(&self, pos: Pos, movement_vec: (isize, isize)) -> (Pos, bool)
    {
        use std::cmp::{min, max};
        let boundaries = self.size();
        let boundaries = (boundaries.x as isize, boundaries.y as isize);
        let mut new_pos_x = pos.x as isize + movement_vec.0;
        let mut new_pos_y = pos.y as isize + movement_vec.1;
        let mut wall_hit = false;

        if new_pos_x < 0 {
            new_pos_x = 0;
            wall_hit = true;
        } else if new_pos_x >= boundaries.0 {
            new_pos_x = boundaries.0 - 1;
            wall_hit = true;
        }
        if new_pos_y < 0 {
            new_pos_y = 0;
            wall_hit = true;
        } else if new_pos_y >= boundaries.1 {
            new_pos_y = boundaries.1 - 1;
            wall_hit = true;
        }

//        new_pos_y = max(0, min(boundaries.0 as isize -1, new_pos_y));
//        new_pos_x = max(0, min(boundaries.1 as isize -1, new_pos_x));

        (Pos{x: new_pos_x as usize, y: new_pos_y as usize}, wall_hit)
    }

    pub fn iter(&self) -> EnvIter {
        EnvIter::new(self.size())
    }

    pub fn transition(&self, pos: Pos, movement: Movement) -> (Pos, &Cell)
    {
        let options = &self.transition[&(pos, movement)];
        let r: f32 = rand::random();
        let mut tot_p = 0.0;
        for (new_pos, p) in options {
            tot_p += p;
            if tot_p > r {
                let target_cell = &self.map[new_pos.x][new_pos.y];
                return (*new_pos, target_cell);
            }
        }
        panic!("Illegal state: Invalid transition map for {:?} {:?}", pos, movement);

    }

    pub fn policy_iteration(&self, discount: f32, max_delta: f32) -> f32 {
        let mut V = Array2::<f32>::zeros((self.size().x, self.size().y));
        let mut policy = *DetPolicy::new();
        policy.initialize(&self);
        for i in 1..20 {
            println!("Policy iteration loop {}", i);
            self.value_iterate_pol(Some(&policy), & mut V, discount, max_delta);
            for pos in self.iter() {
                let (_, _, best_action) = self.inner(&pos, Some(&policy), &V, discount);
                policy.policy.insert(pos, best_action);
            }
            println!("{:?}", policy.policy);
        }
        V[[self.start.x, self.start.y]]
    }

    fn value_iterate_pol<P: Policy>(&self, policy: Option<&P>, V: &mut Array2<f32>, discount: f32, max_delta: f32) {
        let mut delta = 1.0;
        let mut count_i = 1;
        while delta > max_delta {
            println!("Iteration {}", count_i);
            delta = 0.0;
            for pos in self.iter() {
                // Only update non-final states
                if let Cell::Ice(r) = self.map[pos.x][pos.y] {
                    let (value, max_Q, _) = self.inner(&pos, policy, V, discount);
                    let new_value = match policy {
                        Some(_) => value,
                        None => max_Q,
                    };
                    delta = delta.max(new_value - V[[pos.x, pos.y]]);
                    V[[pos.x, pos.y]] = new_value;
                    println!("Value ({}, {}): {}", pos.x, pos.y, new_value);
                }
            }
            count_i += 1;
        }
    }

    fn inner<P: Policy>(&self, pos: &Pos, policy: Option<&P>, V: &Array2<f32>, discount: f32) -> (f32, f32, Movement) {
        let actions = [Movement::Up, Movement::Down, Movement::Right, Movement::Left];
        let mut best_action = Movement::Up;
        let mut best_value = -std::f32::INFINITY;
        let action_values = actions.iter().map(|a| -> f32 {
            let p_policy = match policy {
                Some(p) => (&p).prob(&self, *pos, a),
                None => 0.25
            };

            let sum_of_poss: f32 = self.transition[&(*pos, *a)].iter().map(|pos2| -> f32{
                let (new_pos, p) = pos2;
                let reward = self.reward(new_pos) as f32;
                // \sum_{s'} T(s, a, s') * [R(s, a, s') + \gamma V(s')]
                p * (reward + discount * V[[new_pos.x, new_pos.y]])
            }).sum();

            if sum_of_poss > best_value {
                best_value = sum_of_poss;
                best_action = *a;
            }
            // \pi(s, a) * \sum_{s'} T(s, a, s') * [R(s, a, s') + \gamma V(s')]
            p_policy * sum_of_poss
        });
        (action_values.sum(), best_value, best_action)
    }

    // Evaluate policy, or do value iteration by passing policy = None.
    pub fn evaluate_policy<P: Policy>(&self, policy: &P, discount: f32, max_delta: f32) -> f32 {
        let mut V = Array2::<f32>::zeros((self.size().x, self.size().y));
        self.value_iterate_pol(Some(policy), & mut V, discount, max_delta);

        V[[self.start.x, self.start.y]]
    }

    pub fn value_iteration(&self, discount: f32, max_delta: f32) -> Array2<f32> {
        let mut V = Array2::<f32>::zeros((self.size().x, self.size().y));
        self.value_iterate_pol(None::<&RandomPolicy>, & mut V, discount, max_delta);

        V
    }
}

pub struct EnvIter {
    currx: usize,
    curry: usize,
    first: bool,
    size: Pos,
}
impl EnvIter{
    fn new(size: Pos) -> EnvIter{
        EnvIter{
            size,
            currx:0,
            curry:0,
            first:true,
        }
    }
}
impl Iterator for EnvIter {
    type Item = Pos;

    fn next(&mut self) -> Option<Pos> {
        if self.first {
            self.first = false;
            return Some(Pos{x:0, y:0});
        }
        self.curry += 1;
        if self.curry == self.size.y {
            self.curry = 0;
            self.currx += 1;
            if self.currx == self.size.x {
                return None;
            }
        }
        Some(Pos{x:self.currx, y: self.curry})
    }
}

pub struct Agent{
    pos: Pos,
    reward: (i32),
}

impl Agent {
    pub fn new(env: &Env) -> Self
    {
        Self {
            pos: env.start_pos(),
            reward: 0,
        }
    }

    pub fn r#move(&mut self, env: &Env, movement: Movement) -> Option<i32>
    {
        let (new_pos, cell) = env.transition(self.pos, movement);
        self.reward += cell.reward();
        self.pos = new_pos;

        match cell {
            Cell::Final(_) => Some(self.reward),
            _ => None,
        }
    }
}

pub trait Policy
{
    fn new() -> Box<Self>;
    fn solve(&self, env: &Env, agent:&mut Agent) -> i32;
    fn prob(&self, env:&Env, pos: Pos, movement: &Movement) -> f32;
}

struct RandomPolicy {
}

impl Policy for RandomPolicy
{
    fn new() -> Box<Self>
    {
//        let agent = Agent::new(&env);
        Box::new(Self { })
    }

    fn solve(&self, env: &Env, agent:&mut Agent) -> i32 {
        let mut s: Option<i32> = None;
        while s.is_none() {
            let movement = rand::random();
            s = agent.r#move(env, movement);
            print!("{:?} => {:?} {:?} \n", movement, s, agent.pos)
        }
        s.unwrap()
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
        let mut s: Option<i32> = None;

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
                s = agent.r#move(env, movement);
                print!("{:?} => {:?} {:?} \n", movement, s, agent.pos);

                if let Some(result) = s {
                    return result;
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
        for pos in env.iter() {
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
        let mut s: Option<i32> = None;
        let actions = [Movement::Up, Movement::Down, Movement::Right, Movement::Left];
        while s.is_none() {
            let r: f32 = rand::random();
            let mut tot_p = 0.0;
            for a in actions.iter() {
                tot_p += self.policy[&(agent.pos, *a)];
                if tot_p > r {
                    s = agent.r#move(env, *a);
                    print!("{:?} => {:?} {:?} \n", a, s, agent.pos);
                    continue;
                }
            }
        }
        s.unwrap()
    }

    fn prob(&self, env: &Env, pos: Pos, movement: &Movement) -> f32 {
        self.policy[&(pos, *movement)]
    }
}

// Represents deterministic policy
struct DetPolicy {
    policy: HashMap<Pos, Movement>,
}

impl DetPolicy{

    // Initializes the deterministic policy to always go up
    fn initialize(& mut self, env: &Env) {
        for pos in env.iter() {
            self.policy.insert(pos, Movement::Up);
        }
    }
}

impl Policy for DetPolicy{
    fn new() -> Box<Self> {
        Box::new(Self {
            policy: HashMap::new(),
        })
    }

    fn solve(&self, env: &Env, agent: &mut Agent) -> i32 {
        let mut s: Option<i32> = None;
        while s.is_none() {
            let a = self.policy[&agent.pos];
            s = agent.r#move(env, a);
            print!("{:?} => {:?} {:?} \n", a, s, agent.pos)
        }
        s.unwrap()
    }

    fn prob(&self, env: &Env, pos: Pos, movement: &Movement) -> f32 {
        if self.policy[&pos] == *movement {1.0} else {0.0}
    }
}

fn main() {
    println!("Hello, world!");

    let env = Env::new();
    let mut agent = Agent::new(&env);
    let policy = RandomPolicy::new();
//    let policy = HumanControlPolicy::new();
    println!("Evaluation of policy: {}", (&env).evaluate_policy(&*policy, 0.9, 0.001));
    println!("Value iteration: {}", (&env).value_iteration(0.9, 0.001));
    println!("Policy iteration: {}", (&env).policy_iteration(0.9, 0.001));
    let result = policy.solve(&env, & mut agent);

    println!("Finished with result {}", result);
}
