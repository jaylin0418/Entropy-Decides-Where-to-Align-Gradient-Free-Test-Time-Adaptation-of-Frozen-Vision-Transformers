"""The report's tables (CPU; reads paper/report/report.tex and paper/numbers.tex): every bold number is the best of its row, of
its column, or of its row within each column group (the groups are the \\cmidrule spans of the header), whichever the caption
states ("Bold: the best of the row" / "Bold: the best in the column" / "Bold: the best of the row within each group"), a tied
best is bold too, and a table whose caption states no rule has no bold number. The caption "Bold: the best of the row among ours,
underlined among the ETA columns" marks the first column group in bold and the second with an underline."""
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
NUM = re.compile(r'[+-]?\d+(?:\.\d+)?$')


def numbers():
    return dict(re.findall(r'\\csname pn@(.+?)\\endcsname\{(.*?)\}', (ROOT / 'paper' / 'numbers.tex').read_text()))


def tables():
    tex = (ROOT / 'paper' / 'report' / 'report.tex').read_text()
    for m in re.finditer(r'\\caption\{(.*?)\}\s*\\label\{(tab:[^}]+)\}.*?\\begin\{tabular\}\{[^}]*\}\n(.*?)\\end\{tabular\}', tex, re.S):
        yield m.group(2), m.group(1), m.group(3)


def value(cell, num):
    """The number in a table cell (None for text, a share with \\%, or a dash) and whether the cell is bold."""
    bold = '\\textbf{' in cell or '\\underline{' in cell
    c = re.sub(r'\\pn\{([^}]+)\}', lambda m: num[m.group(1)], cell)
    c = c.replace('\\textbf{', '').replace('\\boldmath', '').replace('\\underline{', '').replace('\\ensuremath{-}', '-').replace('$-$', '-').replace('}', '').replace('$', '').strip()
    if '\\%' in cell or c in ('', '--'):
        return None, bold
    return (float(c) if NUM.match(c) else None), bold


def grid(body, num):
    rows = []
    for line in body.split('\\\\'):
        if '&' not in line or '\\multicolumn' in line or '\\cmidrule' in line:
            continue
        rows.append([value(c, num) for c in line.strip().split('&')])
    return rows


def test_bold_cells_are_the_best_of_their_row_or_column():
    num = numbers(); checked = 0; labels = set()
    for label, caption, body in tables():
        labels.add(label)
        rows = grid(body, num)
        rule = caption.split('Bold: the best')[1][:40] if 'Bold: the best' in caption else None
        groups = [(int(a) - 1, int(b)) for a, b in re.findall(r'\\cmidrule\(lr\)\{(\d+)-(\d+)\}', body)]
        if rule is not None and 'column' in rule:
            for j in range(max(len(r) for r in rows)):
                col = [r[j] for r in rows if j < len(r) and r[j][0] is not None]
                if not any(b for _, b in col):
                    continue
                best = max(v for v, _ in col)
                assert all(v == best for v, b in col if b), (label, 'column', j, 'a bold number is not the best')
                assert all(b for v, b in col if v == best), (label, 'column', j, 'a tied best is not bold')
                checked += 1
        elif rule is not None:
            if 'among ours' in rule:
                assert len(groups) == 2 and 'underlined among the ETA columns' in caption, label
                for lo, hi in groups:                                  # bold only in the first group, underline only in the second
                    mark = '\\textbf{' if lo == groups[0][0] else '\\underline{'
                    assert all(mark in c for c in re.findall(r'\\(?:textbf|underline)\{\\pn\{[^}]*\}\}', ' '.join(l.split('&')[k] for l in body.split('\\\\') if l.count('&') >= hi - 1 for k in range(lo, hi)))), (label, 'a mark in the wrong group')
            spans = groups if 'within each group' in rule or 'among ours' in rule else [(0, 10 ** 6)]
            assert spans, (label, 'a group rule without column groups in the header')
            for i, r in enumerate(rows):
                for lo, hi in spans:
                    vals = [(v, b) for v, b in r[lo:hi] if v is not None]
                    if not any(b for _, b in vals):
                        continue
                    best = max(v for v, _ in vals)
                    assert all(v == best for v, b in vals if b), (label, 'row', i, 'a bold number is not the best')
                    assert all(b for v, b in vals if v == best), (label, 'row', i, 'a tied best is not bold')
                    checked += 1
        else:
            assert not any(b for r in rows for v, b in r if v is not None), (label, 'a bold number without a rule in the caption')
    assert {'tab:exit', 'tab:calibsize', 'tab:across', 'tab:natural', 'tab:ctl', 'tab:others'} <= labels
    assert checked >= 20
