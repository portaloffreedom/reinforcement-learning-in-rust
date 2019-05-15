use rand::prelude::*;

struct RandomExplorationStrategy {}
struct EpsilonGreedyExplorationStrategy {}
struct SoftMaxExplorationStrategy {}

trait ExplorationStrategy {
    fn next_state();
}

fn max_value_next_action() {}
fn random_or_max_next_action<T: ExplorationStrategy>(rng: &mut ThreadRng, probabiltity_exploration: f32) {
    if rng.gen::<f32>() < probabiltity_exploration {
        T::next_state()
    } else {
        max_value_next_action()
    }
}

struct QLearningActionSelector {
    rng: ThreadRng,
}
struct SARSAActionSelector {
    rng: ThreadRng,
}

impl QLearningActionSelector {
    pub fn new() -> Self {
        Self {
            rng: thread_rng(),
        }
    }
}

impl SARSAActionSelector {
    pub fn new() -> Self {
        Self {
            rng: thread_rng(),
        }
    }
}

trait ActionSelector<T: ExplorationStrategy> {
    fn take_action(&mut self);
    fn predict_action(&mut self);
}

impl<T: ExplorationStrategy> ActionSelector<T> for QLearningActionSelector {
    fn take_action(&mut self)
    { random_or_max_next_action::<T>(&mut self.rng, 0.5) }

    fn predict_action(&mut self)
    { max_value_next_action() }
}

impl<T: ExplorationStrategy> ActionSelector<T> for SARSAActionSelector {
    fn take_action(&mut self)
    { random_or_max_next_action::<T>(&mut self.rng, 0.5) }

    fn predict_action(&mut self)
    { random_or_max_next_action::<T>(&mut self.rng, 0.5) }
}
