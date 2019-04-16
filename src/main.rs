extern crate rand;

use std::io::{self, Read};
use std::io::prelude::*;
use std::vec::Vec;
use rand::{
    distributions::{Distribution, Standard},
    Rng,
};

#[derive(Debug)]
enum Cell {
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
#[derive(Debug, Copy, Clone)]
enum Movement {
    Up,
    Right,
    Down,
    Left,
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

struct Env {
    map: Vec<Vec<Cell>>,
    start: (usize, usize),
}

impl Env {
    pub fn new() -> Self
    {
        Self {
            map: vec![
                vec![Cell::Ice(0), Cell::Ice(0),     Cell::Ice(0),     Cell::Final(100)],
                vec![Cell::Ice(0), Cell::Final(-10), Cell::Ice(0),     Cell::Final(-10)],
                vec![Cell::Ice(0), Cell::Ice(0),     Cell::Ice(20),    Cell::Final(-10)],
                vec![Cell::Ice(0), Cell::Final(-10), Cell::Final(-10), Cell::Final(-10)],
            ],
            start: (3, 0)
        }
    }

    pub fn size(&self) -> (usize, usize)
    {
        (self.map.len(), self.map[0].len())
    }

    pub fn start_pos(&self) -> (usize, usize) { self.start }

    fn check_movement(&self, pos: (usize, usize), movement_vec: (isize, isize)) -> ((usize, usize), bool)
    {
        use std::cmp::{min,max};
        let boundaries = self.size();
        let boundaries =  (boundaries.0 as isize, boundaries.1 as isize);
        let mut new_pos_y = pos.0 as isize + movement_vec.0;
        let mut new_pos_x = pos.1 as isize + movement_vec.1;
        let mut wall_hit = false;

        if new_pos_x < 0 {
            new_pos_x = 0;
            wall_hit = true;
        } else if new_pos_x >= boundaries.0 {
            new_pos_x = boundaries.0 -1;
            wall_hit = true;
        }
        if new_pos_y < 0 {
            new_pos_y = 0;
            wall_hit = true;
        } else if new_pos_y >= boundaries.1 {
            new_pos_y = boundaries.1 -1;
            wall_hit = true;
        }

//        new_pos_y = max(0, min(boundaries.0 as isize -1, new_pos_y));
//        new_pos_x = max(0, min(boundaries.1 as isize -1, new_pos_x));

        ((new_pos_y as usize, new_pos_x as usize), wall_hit)
    }

    pub fn transition(&self, pos: (usize, usize), movement: Movement) -> ((usize, usize), &Cell)
    {
        let mut movement_vec = movement.into_vector();
        let boundaries = self.size();
        let (mut new_pos, wall_hit) = self.check_movement(pos, movement_vec);
        let mut target_cell = &self.map[new_pos.0][new_pos.1];

        match target_cell {
            Cell::Ice(_) => {
                let r: f32 = rand::random();
                if r > 0.95 {
                    print!("I'm sliding! ");

                    let mut wall_hit = false;
                    while !wall_hit {
                        let (_new_pos, _wall_hit) = self.check_movement(new_pos, movement_vec);
                        new_pos = _new_pos;
                        wall_hit = _wall_hit;
                    }
                    target_cell = &self.map[new_pos.0][new_pos.1];
                }
            }
            _ => {},
        }


        (new_pos, target_cell)
    }
}

struct Agent<'env> {
    env: &'env Env,
    pos: (usize, usize),
    reward: (i32),
}

impl<'env> Agent<'env> {
    pub fn new(env: &'env Env) -> Self
    {
        Self {
            env,
            pos: env.start_pos(),
            reward: 0,
        }
    }

    pub fn r#move(&mut self, movement: Movement) -> Option<i32>
    {
        let (new_pos, cell) = self.env.transition(self.pos, movement);
        self.reward += cell.reward();
        self.pos = new_pos;

        match cell {
            Cell::Final(_) => Some(self.reward),
            _ => None,
        }
    }
}

fn main() {
    println!("Hello, world!");

    let env = Env::new();
    let mut agent = Agent::new(&env);
    let mut s: Option<i32> = None;
//    while s.is_none() {
//        let movement = rand::random();
//        s = agent.r#move(movement);
//        print!("{:?} => {:?} {:?} \n", movement, s, agent.pos)
//    }

    let stdin = io::stdin();
    for line in stdin.lock().lines() {
        let line = line.unwrap();
        println!("read {}", line);
        let movement = match line.as_str() {
            "w" => Movement::Up,
            "s" => Movement::Down,
            "a" => Movement::Left,
            "d" => Movement::Right,
            _ => panic!("wrong input"),
        };
        s = agent.r#move(movement);
        print!("{:?} => {:?} {:?} \n", movement, s, agent.pos);

        if let Some(result) = s {
            println!("Finished with resulut {}", result);
            break
        }
    }
}
