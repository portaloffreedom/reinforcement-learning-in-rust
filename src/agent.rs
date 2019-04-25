use crate::environment::{
    Pos,
    Cell,
    Movement,
    Env,
};


pub struct Agent{
    pub pos: Pos,
    pub reward: (i32),
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