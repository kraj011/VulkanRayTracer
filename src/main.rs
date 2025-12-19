use crate::engine::Engine;

mod engine;
mod state;
mod vertex;

fn main() {
    let engine = Engine::setup();
    engine.run();
}
