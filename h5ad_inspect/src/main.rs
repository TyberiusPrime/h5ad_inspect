use std::env;
use std::process;

use hdf5_metno::types::TypeDescriptor;

// Used to read index from old-format h5ad where obs/var is a compound dataset
#[derive(hdf5_metno::H5Type, Clone)]
#[repr(C)]
struct CompoundIndex {
    index: hdf5_metno::types::VarLenUnicode,
}

fn main() {
    let args: Vec<String> = env::args().collect();
    if args.len() != 3 {
        eprintln!("Usage: h5ad-inspect <filename> obs|var|uns|obsm|layers|obs_index|var_index");
        process::exit(1);
    }

    let sections = [
        "obs",
        "var",
        "uns",
        "obsm",
        "layers",
        "obs_index",
        "var_index",
    ];
    let section_arg = args[1..].iter().find(|a| sections.contains(&a.as_str()));
    let section = match section_arg {
        Some(s) => s.as_str(),
        None => {
            eprintln!(
                "Error: section must be one of obs, var, uns, obsm, layers, obs_index, var_index"
            );
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
            let loc_type = match file.loc_type_by_name(section) {
                Ok(t) => t,
                Err(e) => {
                    eprintln!("Error: cannot locate '{}': {}", section, e);
                    process::exit(1);
                }
            };
            if loc_type == hdf5_metno::LocationType::Group {
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
            } else {
                let ds = match file.dataset(section) {
                    Ok(d) => d,
                    Err(e) => {
                        eprintln!("Error: dataset '{}' not found: {}", section, e);
                        process::exit(1);
                    }
                };
                let dtype = match ds.dtype() {
                    Ok(dt) => dt,
                    Err(e) => {
                        eprintln!("Error reading dtype of '{}': {}", section, e);
                        process::exit(1);
                    }
                };
                let desc = match dtype.to_descriptor() {
                    Ok(d) => d,
                    Err(e) => {
                        eprintln!("Error converting dtype of '{}': {}", section, e);
                        process::exit(1);
                    }
                };
                match desc {
                    TypeDescriptor::Compound(ct) => {
                        let mut names: Vec<String> = ct
                            .fields
                            .iter()
                            .map(|f| f.name.clone())
                            .filter(|n| n != "index")
                            .collect();
                        names.sort_unstable();
                        for name in names {
                            println!("{}", name);
                        }
                    }
                    _ => {
                        eprintln!("Error: '{}' is not a compound dataset", section);
                        process::exit(1);
                    }
                }
            }
        }
        "obs_index" | "var_index" => {
            let group_name = if section == "obs_index" { "obs" } else { "var" };
            let loc_type = match file.loc_type_by_name(group_name) {
                Ok(t) => t,
                Err(e) => {
                    eprintln!("Error: cannot locate '{}': {}", group_name, e);
                    process::exit(1);
                }
            };
            if loc_type == hdf5_metno::LocationType::Group {
                // The _index attribute on the group names the index dataset
                let group = match file.group(group_name) {
                    Ok(g) => g,
                    Err(e) => {
                        eprintln!("Error opening group '{}': {}", group_name, e);
                        process::exit(1);
                    }
                };
                let index_name = match group.attr("_index") {
                    Ok(attr) => match attr
                        .read_scalar::<hdf5_metno::types::VarLenUnicode>()
                    {
                        Ok(v) => v.as_str().to_string(),
                        Err(_) => "_index".to_string(),
                    },
                    Err(_) => "_index".to_string(),
                };
                let ds = match file.dataset(&format!("{}/{}", group_name, index_name)) {
                    Ok(d) => d,
                    Err(e) => {
                        eprintln!("Error: {}/{}: {}", group_name, index_name, e);
                        process::exit(1);
                    }
                };
                let arr = match ds.read_1d::<hdf5_metno::types::VarLenUnicode>() {
                    Ok(a) => a,
                    Err(e) => {
                        eprintln!("Error reading {group_name}/{index_name}: {e}");
                        process::exit(1);
                    }
                };
                let mut values: Vec<&str> = arr.iter().map(|v| v.as_str()).collect();
                values.sort_unstable();
                for v in values {
                    println!("{}", v);
                }
            } else {
                let ds = match file.dataset(group_name) {
                    Ok(d) => d,
                    Err(e) => {
                        eprintln!("Error: dataset '{}' not found: {}", group_name, e);
                        process::exit(1);
                    }
                };
                let arr = match ds.read_1d::<CompoundIndex>() {
                    Ok(a) => a,
                    Err(e) => {
                        eprintln!("Error reading compound index '{}': {}", group_name, e);
                        process::exit(1);
                    }
                };
                let mut values: Vec<String> =
                    arr.iter().map(|row| row.index.as_str().to_string()).collect();
                values.sort_unstable();
                for v in values {
                    println!("{}", v);
                }
            }
        }
        _ => {
            eprintln!(
                "Error: section must be one of obs, var, uns, obsm, layers, obs_index, var_index"
            );
            process::exit(1);
        }
    }
}
