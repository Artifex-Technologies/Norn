//! norn-bench MODEL.gguf TOKENIZER.json [REQUESTS.jsonl]: answers requests in jev's wire format, one JSON per line, timing
//! each. Without a file: the SAR-88 latency case, a text of about 200 tokens with three questions (yes/no, choice, score),
//! 3 warm-up runs then 20 timed, median and spread printed.
use std::time::Instant;

fn main() {
    let a: Vec<String> = std::env::args().skip(1).collect();
    if a.len() < 2 {
        eprintln!("usage: norn-bench MODEL.gguf TOKENIZER.json [REQUESTS.jsonl]");
        std::process::exit(2);
    }
    let t = Instant::now();
    let engine = norn::Engine::load(a[0].as_ref(), a[1].as_ref()).unwrap_or_else(|e| {
        eprintln!("norn-bench: {e}");
        std::process::exit(1)
    });
    eprintln!("loaded in {:.0} ms", t.elapsed().as_secs_f64() * 1000.0);
    if let Some(file) = a.get(2) {
        for line in std::fs::read_to_string(file).expect("the requests").lines().filter(|l| !l.trim().is_empty()) {
            let req: serde_json::Value = serde_json::from_str(line).expect("a JSON request per line");
            let t = Instant::now();
            let ans = norn::answer(&engine, &req).expect("an answer");
            let ms = t.elapsed().as_secs_f64() * 1000.0;
            println!("{}", serde_json::json!({"ms": ms, "response": ans}));
        }
        return;
    }
    let text = "Customer message, received Tuesday morning. I ordered a wireless keyboard and a matching mouse three weeks ago, \
        order number 4471. The keyboard arrived on time and works well, but the mouse never came. The tracking page has said \
        'label created' for nineteen days now. I wrote to your support address twice, on the 3rd and again on the 9th, and \
        nobody has answered either message. I was charged for both items on the day I ordered. I am not asking for anything \
        unreasonable: either send the mouse this week or refund what I paid for it. If I do not hear back by Friday I will \
        dispute the charge with my bank and leave a review describing how this was handled. I have bought from you for \
        years and this is the first time anything like this has happened, which is why I am so disappointed.";
    let req = serde_json::json!({
        "state": text,
        "questions": {
            "refund": {"type": "noul", "instructions": "Our policy refunds items not delivered within 14 days. Should this customer get a refund for the mouse?"},
            "team": {"type": "choice", "instructions": "Which team should handle this message?",
                     "criteria": {"billing": "payments, refunds, charges", "shipping": "deliveries, missing parcels, tracking", "tech": "a product that does not work"}},
            "anger": {"type": "score", "instructions": "How angry is the customer?", "criteria": ["calm", "annoyed", "angry", "furious"]}
        }
    });
    let tokens = engine.count(&format!("<s>{}", norn::render_state(&req["state"]))).unwrap_or(0);
    for _ in 0..3 {
        norn::answer(&engine, &req).expect("an answer");
    }
    let mut times = vec![];
    let mut last = serde_json::Value::Null;
    for _ in 0..20 {
        let t = Instant::now();
        last = norn::answer(&engine, &req).expect("an answer");
        times.push(t.elapsed().as_secs_f64() * 1000.0);
    }
    times.sort_by(f64::total_cmp);
    println!(
        "text {tokens} tokens, 3 questions (8 yes/no prompts): median {:.1} ms, fastest {:.1}, slowest {:.1} (20 runs, {} threads)",
        times[10],
        times[0],
        times[19],
        std::thread::available_parallelism().map(|n| n.get()).unwrap_or(1)
    );
    println!("{last}");
}
