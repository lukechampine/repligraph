use std::collections::BTreeMap;
use std::env;
use std::fs;
use std::process::exit;

fn main() {
    let args: Vec<String> = env::args().collect();
    if args.len() != 3 {
        eprintln!("usage: wordfreq <input> <output>");
        exit(2);
    }
    let text = match fs::read_to_string(&args[1]) {
        Ok(t) => t,
        Err(e) => {
            eprintln!("wordfreq: {}: {}", args[1], e);
            exit(1);
        }
    };
    let mut counts: BTreeMap<String, u64> = BTreeMap::new();
    for w in text.split_whitespace() {
        let w: String = w
            .chars()
            .filter(|c| c.is_alphanumeric())
            .flat_map(|c| c.to_lowercase())
            .collect();
        if !w.is_empty() {
            *counts.entry(w).or_insert(0) += 1;
        }
    }
    let mut out = String::new();
    for (w, n) in &counts {
        out.push_str(w);
        out.push('\t');
        out.push_str(&n.to_string());
        out.push('\n');
    }
    if let Err(e) = fs::write(&args[2], out) {
        eprintln!("wordfreq: {}: {}", args[2], e);
        exit(1);
    }
    println!("{} distinct words -> {}", counts.len(), args[2]);
}
