"""
Filter a ModEM .dat file by removing impedance points with relative error > THRESHOLD.

Relative error is defined as: err / |Z|, where |Z| = sqrt(Re^2 + Im^2).

Reads:  Mall_Z_no_tippers.dat   (impedance only, no tippers)
Writes: Mall_Z_clean.dat        (filtered)
        removed_points.csv      (audit log of what was removed and why)
"""
import math
import csv
from pathlib import Path
from collections import Counter, defaultdict

DAT_IN  = "/home/claude/depurar/Mall_Z_no_tippers.dat"
DAT_OUT = "/home/claude/depurar/Mall_Z_clean.dat"
CSV_LOG = "/home/claude/depurar/removed_points.csv"

REL_ERR_THRESHOLD = 0.50      # drop points with err/|Z| > 50%
APPLY_TO_COMPS    = ('ZXX', 'ZXY', 'ZYX', 'ZYY')   # all Z components


def main():
    with open(DAT_IN, 'rb') as f:
        raw = f.read()
    has_crlf = b'\r\n' in raw
    eol = '\r\n' if has_crlf else '\n'
    lines = raw.decode('utf-8', errors='replace').splitlines()

    # The header section also contains a "> 58 40" line (n_periods n_sites).
    # We need to update it if any whole period or site disappears.
    # For safety, recompute it after filtering.

    out_lines = []
    removed = []   # list of dicts for the audit log
    kept_by_comp    = Counter()
    removed_by_comp = Counter()
    removed_by_site = Counter()
    sites_kept   = set()
    periods_kept = set()

    # We'll keep header lines verbatim except for the "> n_periods n_sites" line,
    # which we'll patch at the end. Track its index in out_lines so we can rewrite it.
    n_period_site_idx = None
    saw_first_directive = False

    for line in lines:
        s = line.strip()

        # Header lines
        if s.startswith('#') or s.startswith('>'):
            out_lines.append(line)
            # Detect the "> N M" line (two integers): it's the periods+sites count.
            # It comes after "> 39.476669 2.895278" (origin), as the last > line.
            if s.startswith('>') and not s.startswith('> Full') and not 'omega' in s.lower():
                toks = s.lstrip('>').split()
                if len(toks) == 2:
                    try:
                        int(toks[0]); int(toks[1])
                        n_period_site_idx = len(out_lines) - 1
                    except ValueError:
                        pass
            continue

        if not s:
            out_lines.append(line)
            continue

        toks = s.split()
        if len(toks) < 11:
            out_lines.append(line)
            continue

        try:
            T = float(toks[0])
            site = toks[1]
            comp = toks[7]
            re_v = float(toks[8])
            im_v = float(toks[9])
            err  = float(toks[10])
        except ValueError:
            out_lines.append(line)
            continue

        # Decide whether to drop
        Zmod = math.hypot(re_v, im_v)
        rel_err = err / Zmod if Zmod > 0 else float('inf')

        drop = False
        reason = ''
        if comp in APPLY_TO_COMPS and rel_err > REL_ERR_THRESHOLD:
            drop = True
            reason = f'rel_err={rel_err*100:.1f}% > {REL_ERR_THRESHOLD*100:.0f}%'
        if Zmod == 0:
            drop = True
            reason = '|Z|=0 (zero impedance)'

        if drop:
            removed.append({
                'site': site, 'T': T, 'comp': comp,
                'real': re_v, 'imag': im_v, 'err': err,
                'Zmod': Zmod, 'rel_err': rel_err,
                'reason': reason
            })
            removed_by_comp[comp] += 1
            removed_by_site[site] += 1
        else:
            out_lines.append(line)
            kept_by_comp[comp] += 1
            sites_kept.add(site)
            periods_kept.add(round(T, 12))

    # Patch the "> N_periods N_sites" line if present
    if n_period_site_idx is not None:
        new_n = f'> {len(periods_kept)} {len(sites_kept)}'
        # Preserve any trailing CR if original had it (unlikely after splitlines, but safe)
        out_lines[n_period_site_idx] = new_n

    # Strip trailing blank lines
    while out_lines and not out_lines[-1].strip():
        out_lines.pop()

    # Write cleaned file
    Path(DAT_OUT).parent.mkdir(parents=True, exist_ok=True)
    with open(DAT_OUT, 'w', newline='') as f:
        for ln in out_lines:
            f.write(ln + eol)

    # Write audit CSV
    with open(CSV_LOG, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=['site', 'T', 'comp', 'real', 'imag',
                                          'err', 'Zmod', 'rel_err', 'reason'])
        w.writeheader()
        for r in removed:
            w.writerow(r)

    # Summary
    n_total = sum(kept_by_comp.values()) + sum(removed_by_comp.values())
    n_kept  = sum(kept_by_comp.values())
    n_drop  = sum(removed_by_comp.values())
    print(f"Input : {DAT_IN}")
    print(f"Output: {DAT_OUT}")
    print(f"Audit : {CSV_LOG}")
    print()
    print(f"Threshold: drop points with rel_err > {REL_ERR_THRESHOLD*100:.0f}%")
    print(f"Applied to components: {APPLY_TO_COMPS}")
    print()
    print(f"Total points: {n_total}")
    print(f"  Kept   : {n_kept}  ({100*n_kept/n_total:.1f}%)")
    print(f"  Removed: {n_drop}  ({100*n_drop/n_total:.1f}%)")
    print()
    print("By component:")
    for c in APPLY_TO_COMPS:
        kept = kept_by_comp[c]
        rem  = removed_by_comp[c]
        tot  = kept + rem
        print(f"  {c}: {tot} total, kept {kept}, removed {rem}  "
              f"({100*rem/tot:.1f}% if tot else 0%)")
    print()
    print(f"Periods kept: {len(periods_kept)} (originally 58)")
    print(f"Sites   kept: {len(sites_kept)} (originally 40)")
    print()
    print("Stations with most removed points:")
    for site, n in sorted(removed_by_site.items(), key=lambda x: -x[1])[:15]:
        print(f"  {site}: {n} points removed")


if __name__ == '__main__':
    main()
