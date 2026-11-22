//! Norn's engine: jevos, a small decision model (a llama GGUF), answering questions about a text with probabilities, on the
//! CPU. Three kinds of question, in jev's wire format:
//! - `noul`: yes or no, answered with P(yes);
//! - `choice`: one option among several, asked as one yes/no question per option against the others;
//! - `score`: a level on a scale, asked as one "this level or higher?" question per level above the lowest.
//!
//! All the questions about one text read it once: the prompts' common start is run through the model once, and each
//! question goes on from a copy of what it left (the keys and values cached in each layer).
//!
//! The prompt layout, the answer tokens and the way yes/no answers become a choice or a score are jev's (MIT, Federico Elia and
//! Loris Salsi; github.com/feder-cr/jev: src/prompt.hpp, choice.hpp, score.hpp, wire.cpp), ported; the model jevos is MIT,
//! made from MiniCPM5-1B (openbmb, Apache-2.0).
use std::path::Path;

use candle_core::quantized::gguf_file;
use candle_core::{Device, Tensor};
mod llama;
use llama::ModelWeights;
use tokenizers::Tokenizer;

pub type Result<T> = std::result::Result<T, String>;

fn err<E: std::fmt::Display>(e: E) -> String {
    e.to_string()
}

/// The phrasing asked for each option of a choice (jev's CHOICE_PHRASINGS, the one it asks).
pub const CHOICE_PHRASING: &str = "Among the candidates, is it this one?";
/// The phrasing asked for each level of a score (jev's SCORE_PHRASINGS).
pub const SCORE_PHRASING: &str = "Is the answer at the following level or higher?";

/// The model and its tokenizer, loaded once.
pub struct Engine {
    model: ModelWeights,
    tok: Tokenizer,
    yes: u32,
    no: u32,
    device: Device,
}

impl Engine {
    /// Loads jevos from its GGUF and the tokenizer.json beside it.
    pub fn load(gguf: &Path, tokenizer: &Path) -> Result<Engine> {
        let device = Device::Cpu;
        let mut f = std::fs::File::open(gguf).map_err(|e| format!("{}: {e}", gguf.display()))?;
        let content = gguf_file::Content::read(&mut f).map_err(err)?;
        let model = ModelWeights::from_gguf(content, &mut f, &device).map_err(err)?;
        let tok = Tokenizer::from_file(tokenizer).map_err(err)?;
        let one = |d: &str| -> Result<u32> {
            let ids = tok.encode(d, false).map_err(err)?.get_ids().to_vec();
            match ids.as_slice() {
                [id] => Ok(*id),
                _ => Err(format!("the answer \"{d}\" is not one token: {ids:?}")),
            }
        };
        let (no, yes) = (one("0")?, one("1")?);
        Ok(Engine { model, tok, yes, no, device })
    }

    fn tokens(&self, text: &str) -> Result<Vec<u32>> {
        Ok(self.tok.encode(text, false).map_err(err)?.get_ids().to_vec())
    }

    /// The number of tokens a text is.
    pub fn count(&self, text: &str) -> Result<usize> {
        Ok(self.tokens(text)?.len())
    }

    /// P(yes) for each prompt: the logits of "1" and "0" after it, as a sigmoid of their difference. The prompts' common
    /// start is read once.
    pub fn yes_probabilities(&self, prompts: &[String]) -> Result<Vec<f64>> {
        let all: Vec<Vec<u32>> = prompts.iter().map(|p| self.tokens(p)).collect::<Result<_>>()?;
        let Some(first) = all.first() else { return Ok(vec![]) };
        // the common start, leaving at least one token of each prompt to read after it
        let mut common = first.len();
        for t in &all[1..] {
            common = common.min(first.iter().zip(t).take_while(|(a, b)| a == b).count());
        }
        common = common.min(all.iter().map(|t| t.len()).min().unwrap_or(1) - 1);
        let mut shared = self.model.clone();
        if common > 0 {
            let x = Tensor::new(&first[..common], &self.device).map_err(err)?.unsqueeze(0).map_err(err)?;
            shared.forward(&x, 0).map_err(err)?;
        }
        let mut out = Vec::with_capacity(all.len());
        for t in &all {
            let mut m = shared.clone();
            let x = Tensor::new(&t[common..], &self.device).map_err(err)?.unsqueeze(0).map_err(err)?;
            let logits = m.forward(&x, common).map_err(err)?.squeeze(0).map_err(err)?;
            let l: Vec<f32> = logits.to_vec1().map_err(err)?;
            let (y, n) = (l[self.yes as usize] as f64, l[self.no as usize] as f64);
            out.push(1.0 / (1.0 + (n - y).exp()));
        }
        Ok(out)
    }
}

/// The evidence block: a plain text as it is (trimmed), anything else as JSON (indent 1, as Python's json.dumps writes it).
pub fn render_state(state: &serde_json::Value) -> String {
    let body = match state {
        serde_json::Value::String(s) if !s.to_lowercase().contains("</evidence>") => s.trim().to_string(),
        v => py_json_indent1(v),
    };
    format!("<evidence>\n{body}\n</evidence>")
}

/// Python's json.dumps(v, ensure_ascii=False, indent=1).
fn py_json_indent1(v: &serde_json::Value) -> String {
    fn go(v: &serde_json::Value, depth: usize, out: &mut String) {
        let pad = |d: usize| " ".repeat(d);
        match v {
            serde_json::Value::Object(m) if !m.is_empty() => {
                out.push_str("{\n");
                for (i, (k, x)) in m.iter().enumerate() {
                    out.push_str(&pad(depth + 1));
                    out.push_str(&serde_json::to_string(k).unwrap_or_default());
                    out.push_str(": ");
                    go(x, depth + 1, out);
                    if i + 1 < m.len() {
                        out.push(',');
                    }
                    out.push('\n');
                }
                out.push_str(&pad(depth));
                out.push('}');
            }
            serde_json::Value::Array(a) if !a.is_empty() => {
                out.push_str("[\n");
                for (i, x) in a.iter().enumerate() {
                    out.push_str(&pad(depth + 1));
                    go(x, depth + 1, out);
                    if i + 1 < a.len() {
                        out.push(',');
                    }
                    out.push('\n');
                }
                out.push_str(&pad(depth));
                out.push(']');
            }
            x => out.push_str(&serde_json::to_string(x).unwrap_or_default()),
        }
    }
    let mut s = String::new();
    go(v, 0, &mut s);
    s
}

/// A yes/no question's prompt (jev's binary_prompt).
pub fn binary_prompt(state_text: &str, instruction: &str) -> String {
    format!("<s>{state_text}\n\nQuestion: {instruction}\nAnswer:")
}

/// One option of a choice asked against the others (jev's choice_instructions).
pub fn choice_instructions(instructions: &str, options: &[String], i: usize) -> String {
    let mut out = format!("{instructions}\nCandidates:");
    for o in options {
        out.push_str("\n- ");
        out.push_str(o);
    }
    format!("{out}\n{CHOICE_PHRASING}\nCandidate: {}", options[i])
}

/// One level of a score asked as "this level or higher?" (jev's score_instructions).
pub fn score_instructions(instructions: &str, levels: &[String], k: usize) -> String {
    format!("{instructions}\nScale from lowest to highest: {}\n{SCORE_PHRASING}\nLevel: {}", levels.join(" < "), levels[k])
}

/// A choice from each option's P(yes): normalized to sum 1; the first best; confidence (p_max - 1/n) / (1 - 1/n) (jev's choose).
pub fn choose(ps: &[f64]) -> (usize, Vec<f64>, f64) {
    let n = ps.len().max(1);
    let sum: f64 = ps.iter().sum();
    let p: Vec<f64> = ps.iter().map(|x| if sum > 0.0 { x / sum } else { 1.0 / n as f64 }).collect();
    let best = (1..p.len()).fold(0, |b, i| if p[i] > p[b] { i } else { b });
    let u = 1.0 / n as f64;
    let conf = ((p.get(best).copied().unwrap_or(u) - u) / (1.0 - u)).clamp(0.0, 1.0);
    (best, p, conf)
}

/// A score from P(level >= k) for k = 1..n-1: made non-increasing (pool adjacent violators), then a distribution over the
/// levels, the expected level, and confidence 1 - E|i - mode| / the same for a uniform distribution (jev's rate).
pub fn rate(ge: &[f64], n: usize) -> (f64, Vec<f64>, f64) {
    let t = n - 1;
    let mut pools: Vec<(f64, usize)> = vec![];
    for &v in ge {
        pools.push((v, 1));
        while pools.len() > 1 {
            let (a, b) = (pools[pools.len() - 2], pools[pools.len() - 1]);
            if a.0 * (b.1 as f64) < b.0 * (a.1 as f64) {
                pools.pop();
                let last = pools.len() - 1;
                pools[last] = (a.0 + b.0, a.1 + b.1);
            } else {
                break;
            }
        }
    }
    let mut g: Vec<f64> = vec![];
    for (sum, count) in pools {
        g.extend(std::iter::repeat_n((sum / count as f64).clamp(0.0, 1.0), count));
    }
    let p: Vec<f64> =
        (0..n).map(|i| ((if i == 0 { 1.0 } else { g[i - 1] }) - (if i == t { 0.0 } else { g[i] })).max(0.0)).collect();
    let score: f64 = p.iter().enumerate().map(|(i, x)| i as f64 * x).sum();
    let mode = (1..n).fold(0, |m, i| if p[i] > p[m] { i } else { m });
    let middle = t as f64 / 2.0;
    let spread: f64 = p.iter().enumerate().map(|(i, x)| x * (i as f64 - mode as f64).abs()).sum();
    let uniform: f64 = (0..n).map(|i| (i as f64 - middle).abs() / n as f64).sum();
    (score, p, (1.0 - spread / uniform).clamp(0.0, 1.0))
}

/// A request in jev's wire format answered: {"answers": {name: {...}}, "usage": {"input_tokens": n}}.
pub fn answer(engine: &Engine, request: &serde_json::Value) -> Result<serde_json::Value> {
    use serde_json::{json, Value};
    let state = render_state(request.get("state").ok_or("no state")?);
    let questions = request.get("questions").and_then(Value::as_object).ok_or("no questions")?;
    // every prompt, and which question and part it answers
    let mut prompts: Vec<String> = vec![];
    let mut plan: Vec<(String, String, Vec<String>, usize)> = vec![]; // name, kind, option keys or level legends, first prompt
    for (name, q) in questions {
        let kind = q.get("type").and_then(Value::as_str).unwrap_or("noul").to_string();
        let instr = |default: &str| {
            q.get("instructions").and_then(Value::as_str).map(str::trim).filter(|s| !s.is_empty()).unwrap_or(default).to_string()
        };
        let at = prompts.len();
        match kind.as_str() {
            "noul" => {
                prompts.push(binary_prompt(&state, &instr("Is this true?")));
                plan.push((name.clone(), kind, vec![], at));
            }
            "choice" => {
                let crit = q.get("criteria").and_then(Value::as_object).ok_or(format!("{name}: a choice needs criteria"))?;
                let keys: Vec<String> = crit.keys().cloned().collect();
                let texts: Vec<String> = crit
                    .iter()
                    .map(|(k, d)| match d.as_str().map(str::trim) {
                        Some("") | None if d.is_null() => k.clone(),
                        Some("") => k.clone(),
                        Some(s) => format!("{k}: {s}"),
                        None => format!("{k}: {d}"),
                    })
                    .collect();
                let i = instr("Which option fits the evidence best?");
                for j in 0..texts.len() {
                    prompts.push(binary_prompt(&state, &choice_instructions(&i, &texts, j)));
                }
                plan.push((name.clone(), kind, keys, at));
            }
            "score" => {
                let crit = q.get("criteria").and_then(Value::as_array).ok_or(format!("{name}: a score needs criteria"))?;
                let levels: Vec<String> = crit.iter().map(|l| l.as_str().map(|s| s.trim().to_string()).unwrap_or(l.to_string())).collect();
                let i = instr("Which level fits the evidence best?");
                for k in 1..levels.len() {
                    prompts.push(binary_prompt(&state, &score_instructions(&i, &levels, k)));
                }
                plan.push((name.clone(), kind, levels, at));
            }
            k => return Err(format!("{name}: no question type {k}")),
        }
    }
    let ps = engine.yes_probabilities(&prompts)?;
    let mut answers = serde_json::Map::new();
    for (name, kind, parts, at) in plan {
        let a = match kind.as_str() {
            "noul" => json!({"type": "noul", "noul": ps[at]}),
            "choice" => {
                let (best, p, conf) = choose(&ps[at..at + parts.len()]);
                let probs: serde_json::Map<String, Value> = parts.iter().zip(&p).map(|(k, x)| (k.clone(), json!(x))).collect();
                json!({"type": "choice", "choice": parts[best], "probabilities": probs, "confidence": conf})
            }
            _ => {
                let n = parts.len();
                let (score, p, conf) = rate(&ps[at..at + n - 1], n);
                json!({"type": "score", "score": score, "probabilities": p, "confidence": conf})
            }
        };
        answers.insert(name, a);
    }
    let input_tokens = engine.count(&format!("<s>{state}"))?;
    Ok(json!({"model": "jevos-v4", "answers": answers, "usage": {"input_tokens": input_tokens, "output_tokens": 0}}))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn choices_and_scores_are_made_from_yes_no_answers_as_jev_makes_them() {
        let (best, p, conf) = choose(&[0.9, 0.3, 0.3]);
        assert_eq!(best, 0);
        assert!((p.iter().sum::<f64>() - 1.0).abs() < 1e-12);
        assert!((conf - ((0.6 - 1.0 / 3.0) / (2.0 / 3.0))).abs() < 1e-9);
        // P(>= k) out of order is pooled: 0.4 then 0.6 become 0.5 and 0.5
        let (_, p, _) = rate(&[0.4, 0.6], 3);
        assert!((p[0] - 0.5).abs() < 1e-12 && p[1].abs() < 1e-12 && (p[2] - 0.5).abs() < 1e-12, "{p:?}");
        let (score, _, _) = rate(&[1.0, 1.0, 0.0], 4);
        assert!((score - 2.0).abs() < 1e-12);
        assert_eq!(render_state(&serde_json::json!("  plain text ")), "<evidence>\nplain text\n</evidence>");
        assert_eq!(render_state(&serde_json::json!({"a": 1})), "<evidence>\n{\n \"a\": 1\n}\n</evidence>");
    }
}
