use ray_tracer::parser::Parser;

fn main() {
    // let engine = Engine::setup();
    // engine.run();
    let mut parser = Parser::new();
    let _ = parser.parse(&"C:\\Users\\haxan\\Desktop\\scenes\\cornell_box.usdc".to_string());
    dbg!(parser.meshes);
    // let _ = test(&"C:\\Users\\haxan\\Desktop\\scenes\\cornell_box.usdc".to_string());
}
