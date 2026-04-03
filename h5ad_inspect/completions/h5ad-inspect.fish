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
    for kw in obs var uns obsm layers obs_index var_index export
        contains -- $kw $toks; and return 1
    end
    return 0
end

# True when "export" is already in the command line.
function __h5ad_has_export
    contains -- export (commandline -opc)
end

# Print the token immediately following "export" (the sub-command), if any.
function __h5ad_export_subcmd
    set -l toks (commandline -opc)
    set -l take_next 0
    for tok in $toks
        if test $take_next -eq 1
            echo $tok
            return 0
        end
        if test "$tok" = export
            set take_next 1
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
    -a obs_index -d 'Show obs index (sorted)'
complete -c h5ad-inspect -n '__h5ad_has_file; and __h5ad_needs_section' -f \
    -a var_index -d 'Show var index (sorted)'
complete -c h5ad-inspect -n '__h5ad_has_file; and __h5ad_needs_section' -f \
    -a export    -d 'Export values to stdout'

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

# ── Dynamic name completions ──────────────────────────────────────────────────

# After "export obs": complete obs column names from the file.
complete -c h5ad-inspect \
    -n 'set -l __sc (__h5ad_export_subcmd 2>/dev/null); test "$__sc" = obs' -f \
    -a '(__h5ad_run obs)'

# After "export var": complete var column names from the file.
complete -c h5ad-inspect \
    -n 'set -l __sc (__h5ad_export_subcmd 2>/dev/null); test "$__sc" = var' -f \
    -a '(__h5ad_run var)'

# After "export row": complete obs index values (cell IDs) from the file.
complete -c h5ad-inspect \
    -n 'set -l __sc (__h5ad_export_subcmd 2>/dev/null); test "$__sc" = row' -f \
    -a '(__h5ad_run obs_index)'

# After "export column": complete var index values (gene/feature IDs) from the file.
complete -c h5ad-inspect \
    -n 'set -l __sc (__h5ad_export_subcmd 2>/dev/null); test "$__sc" = column' -f \
    -a '(__h5ad_run var_index)'
