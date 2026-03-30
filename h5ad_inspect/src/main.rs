use std::env;
use std::process;

fn main() {
    let args: Vec<String> = env::args().collect();
    if args.len() != 3 {
        eprintln!("Usage: h5ad-inspect <filename> obs|var|uns|obsm");
        process::exit(1);
    }

    let filename = &args[1];
    let section = args[2].as_str();

    match section {
        "obs" | "var" | "uns" | "obsm" => {}
        _ => {
            eprintln!("Error: section must be one of obs, var, uns, obsm");
            process::exit(1);
        }
    }

    let file = match hdf5_metno::File::open(filename) {
        Ok(f) => f,
        Err(e) => {
            eprintln!("Error opening {}: {}", filename, e);
            process::exit(1);
        }
    };

    let group = match file.group(section) {
        Ok(g) => g,
        Err(e) => {
            eprintln!("Error: group '{}' not found in {}: {}", section, filename, e);
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

    // obs and var: exclude internal HDF5 metadata keys
    if section == "obs" || section == "var" {
        names.retain(|n| n != "_index" && n != "__categories");
    }

    names.sort_unstable();

    for name in &names {
        println!("{}", name);
    }
}
