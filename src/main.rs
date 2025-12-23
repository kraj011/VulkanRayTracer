use ray_tracer::{engine::Engine, parser::Parser};

fn main() {
    let engine = Engine::setup();
    let mut parser = Parser::new();
    let success = parser.parse(&"C:\\Users\\haxan\\Desktop\\scenes\\cornell_box.usdc".to_string());

    match success {
        Err(err) => panic!("{}", err),
        Ok(val) => {
            if !val {
                panic!("Parser finished in error state. Please debug,")
            }
        }
    }

    engine.create_buffers(&mut parser);
    engine.run(&parser);
}
