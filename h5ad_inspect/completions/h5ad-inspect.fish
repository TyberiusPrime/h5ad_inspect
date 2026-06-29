# Fish completions for h5ad-inspect
#
# Install (manual): cp h5ad-inspect.fish ~/.config/fish/completions/
# Install (Nix):    built into the package; available automatically.

# ── Helpers ───────────────────────────────────────────────────────────────────

# Return the .h5ad filename from the already-typed tokens.
function __h5ad_get_file
    for tok in (commandline -opc)[2..]
        if string match -q '*.h5ad' -- $tok
            echo $tok
            return 0
        end
    end
    return 1
end

# True when a .h5ad file has already been provided.
function __h5ad_has_file
    __h5ad_get_file >/dev/null 2>&1
end

# True when neither a section keyword nor "export" has been given yet.
function __h5ad_needs_section
    set -l toks (commandline -opc)
    for kw in obs var uns obsm layers shape export
        contains -- $kw $toks; and return 1
    end
    return 0
end

# True when "export" is already in the command line.
function __h5ad_has_export
    contains -- export (commandline -opc)
end

# Print the sub-command following "export" — the first non-flag token after it,
# skipping flags such as --binary / --include-index / --layer <name>.
function __h5ad_export_subcmd
    set -l toks (commandline -opc)
    set -l seen_export 0
    set -l skip_next 0
    for tok in $toks
        if test $skip_next -eq 1
            set skip_next 0
            continue
        end
        if test $seen_export -eq 1
            # --layer takes a value as the following token.
            if test "$tok" = --layer
                set skip_next 1
                continue
            end
            if string match -q -- '-*' $tok
                continue
            end
            echo $tok
            return 0
        end
        if test "$tok" = export
            set seen_export 1
        end
    end
    return 1
end

# True when "export" is present but no sub-command follows it yet.
function __h5ad_needs_export_subcmd
    __h5ad_has_export; and not __h5ad_export_subcmd >/dev/null 2>&1
end

# Run h5ad-inspect against the detected file; output becomes completion candidates.
function __h5ad_run
    set -l file (__h5ad_get_file)
    if test -n "$file"
        h5ad-inspect $file $argv 2>/dev/null
    end
end

# ── File completion ───────────────────────────────────────────────────────────

complete -c h5ad-inspect -n 'not __h5ad_has_file' -F

# ── Section / top-level commands ──────────────────────────────────────────────

complete -c h5ad-inspect -n '__h5ad_has_file; and __h5ad_needs_section' -f \
    -a obs       -d 'List obs columns'
complete -c h5ad-inspect -n '__h5ad_has_file; and __h5ad_needs_section' -f \
    -a var       -d 'List var columns'
complete -c h5ad-inspect -n '__h5ad_has_file; and __h5ad_needs_section' -f \
    -a uns       -d 'List uns keys'
complete -c h5ad-inspect -n '__h5ad_has_file; and __h5ad_needs_section' -f \
    -a obsm      -d 'List obsm keys'
complete -c h5ad-inspect -n '__h5ad_has_file; and __h5ad_needs_section' -f \
    -a layers    -d 'List layer names'
complete -c h5ad-inspect -n '__h5ad_has_file; and __h5ad_needs_section' -f \
    -a shape     -d 'Show n_obs and n_var (cells and genes)'
complete -c h5ad-inspect -n '__h5ad_has_file; and __h5ad_needs_section' -f \
    -a export    -d 'Export values to stdout'

# Inspect-mode flag: sort the listed keys alphabetically (default is file order).
complete -c h5ad-inspect -n '__h5ad_has_file; and not __h5ad_has_export' -f \
    -l sorted    -d 'Sort listed keys alphabetically'

# ── export flags ──────────────────────────────────────────────────────────────

complete -c h5ad-inspect -n '__h5ad_has_export' -f \
    -l include-index -d 'Prefix obs/var values with their index (cell/gene ID)'
complete -c h5ad-inspect -n '__h5ad_has_export' -f \
    -l binary    -d 'Emit numeric output as raw little-endian f64'

# ── export sub-commands ───────────────────────────────────────────────────────

complete -c h5ad-inspect -n '__h5ad_needs_export_subcmd' -f \
    -a obs_index -d 'Export obs index (in order)'
complete -c h5ad-inspect -n '__h5ad_needs_export_subcmd' -f \
    -a var_index -d 'Export var index (in order)'
complete -c h5ad-inspect -n '__h5ad_needs_export_subcmd' -f \
    -a obs       -d 'Export an obs column'
complete -c h5ad-inspect -n '__h5ad_needs_export_subcmd' -f \
    -a var       -d 'Export a var column'
complete -c h5ad-inspect -n '__h5ad_needs_export_subcmd' -f \
    -a row       -d 'Export X row by obs ID'
complete -c h5ad-inspect -n '__h5ad_needs_export_subcmd' -f \
    -a column    -d 'Export X column by var ID'
complete -c h5ad-inspect -n '__h5ad_needs_export_subcmd' -f \
    -a obs_categories -d 'Categories of an obs column'
complete -c h5ad-inspect -n '__h5ad_needs_export_subcmd' -f \
    -a var_categories -d 'Categories of a var column'
complete -c h5ad-inspect -n '__h5ad_needs_export_subcmd' -f \
    -a obs_encoding -d 'h5ad encoding of an obs column (JSON)'
complete -c h5ad-inspect -n '__h5ad_needs_export_subcmd' -f \
    -a var_encoding -d 'h5ad encoding of a var column (JSON)'
complete -c h5ad-inspect -n '__h5ad_needs_export_subcmd' -f \
    -a matrix_csr -d 'Full X matrix as NumPy .npz (CSR)'
complete -c h5ad-inspect -n '__h5ad_needs_export_subcmd' -f \
    -a matrix_csc -d 'Full X matrix as NumPy .npz (CSC)'

# ── Dynamic name completions ──────────────────────────────────────────────────

# After "export obs": complete obs column names from the file.
complete -c h5ad-inspect \
    -n 'set -l __sc (__h5ad_export_subcmd 2>/dev/null); test "$__sc" = obs' -f \
    -a '(__h5ad_run obs)'

# After "export var": complete var column names from the file.
complete -c h5ad-inspect \
    -n 'set -l __sc (__h5ad_export_subcmd 2>/dev/null); test "$__sc" = var' -f \
    -a '(__h5ad_run var)'

# After "export obs_categories"/"obs_encoding": complete obs column names.
complete -c h5ad-inspect \
    -n 'set -l __sc (__h5ad_export_subcmd 2>/dev/null); contains -- "$__sc" obs_categories obs_encoding' -f \
    -a '(__h5ad_run obs)'

# After "export var_categories"/"var_encoding": complete var column names.
complete -c h5ad-inspect \
    -n 'set -l __sc (__h5ad_export_subcmd 2>/dev/null); contains -- "$__sc" var_categories var_encoding' -f \
    -a '(__h5ad_run var)'

# After "export row": complete obs index values (cell IDs) from the file.
complete -c h5ad-inspect \
    -n 'set -l __sc (__h5ad_export_subcmd 2>/dev/null); test "$__sc" = row' -f \
    -a '(__h5ad_run export obs_index)'

# After "export column": complete var index values (gene/feature IDs) from the file.
complete -c h5ad-inspect \
    -n 'set -l __sc (__h5ad_export_subcmd 2>/dev/null); test "$__sc" = column' -f \
    -a '(__h5ad_run export var_index)'
