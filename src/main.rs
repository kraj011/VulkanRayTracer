use ray_tracer::{engine::Engine, parser::Parser};
use winit::event_loop::EventLoop;

fn main() {
    let event_loop = EventLoop::new().unwrap();
    let engine = Engine::new(&event_loop);

    let mut parser = Parser::new();
    let success = parser.parse(&".\\scenes\\cornell_box.usdc".to_string());

    match success {
        Err(err) => panic!("{}", err),
        Ok(val) => {
            if !val {
                panic!("Parser finished in error state. Please debug,")
            }
        }
    }

    engine.create_buffers(&mut parser);
    engine.run_rt(event_loop, &parser);
}
