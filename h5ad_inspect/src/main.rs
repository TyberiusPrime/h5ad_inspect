use std::env;
use std::process;

fn main() {
    let args: Vec<String> = env::args().collect();
    if args.len() != 3 {
        eprintln!("Usage: h5ad-inspect <filename> obs|var|uns|obsm|layers|obs_index|var_index");
        process::exit(1);
    }

    let sections = ["obs", "var", "uns", "obsm", "layers", "obs_index", "var_index"];
    let section_arg = args[1..].iter().find(|a| sections.contains(&a.as_str()));
    let section = match section_arg {
        Some(s) => s.as_str(),
        None => {
            eprintln!("Error: section must be one of obs, var, uns, obsm, layers, obs_index, var_index");
            process::exit(1);
        }
    };
    let filename = args[1..].iter().find(|a| a.as_str() != section).unwrap();

    let file = match hdf5_metno::File::open(filename) {
        Ok(f) => f,
        Err(e) => {
            eprintln!("Error opening {}: {}", filename, e);
            process::exit(1);
        }
    };

    match section {
        "obs" | "var" | "uns" | "obsm" | "layers" => {
            let group = match file.group(section) {
                Ok(g) => g,
                Err(e) => {
                    eprintln!("Error: group '{}' not found: {}", section, e);
                    process::exit(1);
                }
            };
            let mut names = match group.member_names() {
                Ok(n) => n,
                Err(e) => {
                    eprintln!("Error reading members of '{}': {}", section, e);
                    process::exit(1);
                }
            };
            if section == "obs" || section == "var" {
                names.retain(|n| n != "_index" && n != "__categories");
            }
            names.sort_unstable();
            for name in names {
                println!("{}", name);
            }
        }
        "obs_index" | "var_index" => {
            let group_name = if section == "obs_index" { "obs" } else { "var" };
            let ds = match file.dataset(&format!("{}/_index", group_name)) {
                Ok(d) => d,
                Err(e) => {
                    eprintln!("Error: {}/{}: {}", group_name, "_index", e);
                    process::exit(1);
                }
            };
            let arr = match ds.read_1d::<hdf5_metno::types::VarLenUnicode>() {
                Ok(a) => a,
                Err(e) => {
                    eprintln!("Error reading {group_name}/_index: {e}");
                    process::exit(1);
                }
            };
            let mut values: Vec<&str> = arr.iter().map(|v| v.as_str()).collect();
            values.sort_unstable();
            for v in values {
                println!("{}", v);
            }
        }
        _ => {
            eprintln!("Error: section must be one of obs, var, uns, obsm, layers, obs_index, var_index");
            process::exit(1);
        }
    }
}
