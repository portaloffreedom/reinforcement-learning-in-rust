use crate::environment::{
    Pos,
    Cell,
    Movement,
    Env,
};


pub struct Agent{
    pub pos: Pos,
    pub cum_reward: (i32),
}

impl Agent {
    pub fn new(env: &Env) -> Self
    {
        Self {
            pos: env.start_pos(),
            cum_reward: 0,
        }
    }

    pub fn r#move(&mut self, env: &Env, movement: Movement) -> i32 {
        let (new_pos, cell) = env.transition(self.pos, movement);
        self.cum_reward += cell.reward();
        self.pos = new_pos;

       cell.reward()
    }
}