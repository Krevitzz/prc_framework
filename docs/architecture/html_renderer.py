# html_renderer.py
from pathlib import Path
from typing import List, Dict, Any
from collections import defaultdict
from .models import (
    ProjectData, ModuleInfo, YamlInfo, UnusedIssue, VariableDetail, FunctionDetail
)

_CSS = """
<link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0-beta3/css/all.min.css">
<style>
@import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@300;400;600;700&family=Syne:wght@400;600;800&display=swap');
:root{
  --bg:#080a0d;--bg2:#0d1117;--bg3:#131920;--border:#1a2030;--border2:#243040;
  --text:#b0bcc8;--dim:#485060;--bright:#dce6f0;
  --blue:#4a9eff;--cyan:#00d4c8;--green:#3dba6a;--amber:#f0a020;
  --red:#e05050;--purple:#9a70f0;--slate:#5a7090;--orange:#e07830;
  --mono:'JetBrains Mono',monospace;

  --const-color: #ffaa33;
  --param-color: #33ccff;
  --global-color: #ffaa33;
  --local-color: #99cc99;
  --unused-color: #ff6666;
  --return-color: #ffaa33;
}
*{box-sizing:border-box;margin:0;padding:0;}
body{background:var(--bg);color:var(--text);font-family:var(--mono);
     font-size:11px;line-height:1.65;padding:28px 32px;max-width:1200px;margin:0 auto;}
h1{font-family:'Syne',sans-serif;font-size:20px;font-weight:800;color:#fff;
   letter-spacing:-.3px;margin-bottom:4px;}
.meta{color:var(--dim);font-size:10px;margin-bottom:28px;}
.meta b{color:var(--text);}
.sl{font-size:9px;font-weight:600;letter-spacing:.18em;text-transform:uppercase;
    color:var(--dim);margin:28px 0 10px;padding-left:8px;
    border-left:2px solid var(--border2);}
.card{background:var(--bg2);border:1px solid var(--border);
      padding:12px 14px;margin-bottom:10px;border-radius:2px;}
.card-title{font-size:10px;font-weight:700;letter-spacing:.08em;text-transform:uppercase;
            color:var(--bright);margin-bottom:8px;display:flex;align-items:center;gap:8px;}
.layer-badge{font-size:9px;padding:1px 6px;background:var(--bg3);
             border:1px solid var(--border2);color:var(--cyan);}
.fn{color:var(--cyan);font-size:10px;margin:2px 0;}
.fn span{color:var(--dim);}
.note{font-size:10px;color:var(--dim);margin-top:4px;}
.grid2{display:grid;grid-template-columns:1fr 1fr;gap:10px;}
.grid3{display:grid;grid-template-columns:1fr 1fr 1fr;gap:10px;}
/* tree pliable */
.tree details {
  margin-left: 0;
}
.tree summary {
  cursor: pointer;
  list-style: none;
}
.tree summary::-webkit-details-marker {
  display: none;
}
.tree summary::before {
  content: '▶';
  color: var(--dim);
  display: inline-block;
  width: 1.2em;
}
.tree details[open] > summary::before {
  content: '▼';
}
.tree .dir {
  display: inline;
  color: var(--blue);
}
.tree .file-py, .tree .file-yaml {
  margin-left: 1.5em;
}
/* fonctions pliables (détails individuels) */
.function-details {
  margin-left: 10px;
  margin-top: 6px;
}
.function-details > summary {
  cursor: pointer;
  list-style: none;
  font-weight: 600;
}
.function-details > summary::-webkit-details-marker {
  display: none;
}
.function-details > summary::before {
  content: '▶';
  color: var(--dim);
  display: inline-block;
  width: 1.2em;
}
.function-details[open] > summary::before {
  content: '▼';
}
/* stats */
.stats{display:flex;gap:24px;margin-bottom:20px;flex-wrap:wrap;}
.stat{background:var(--bg2);border:1px solid var(--border);
      padding:10px 16px;min-width:100px;}
.stat-n{font-size:20px;font-weight:700;color:var(--bright);}
.stat-l{font-size:9px;color:var(--dim);margin-top:2px;letter-spacing:.06em;}
.stat-red .stat-n{color:var(--red);}
.stat-amber .stat-n{color:var(--amber);}
/* tables pour constantes */
table{width:100%;border-collapse:collapse;font-size:10px;margin-top:6px;}
th{color:var(--dim);font-weight:600;text-align:left;padding:4px 8px;
   border-bottom:1px solid var(--border2);letter-spacing:.06em;font-size:9px;}
td{padding:4px 8px;border-bottom:1px solid var(--border);vertical-align:top;}
td.obj{color:var(--cyan);}
td.mod{color:var(--blue);}
td.ctx{color:var(--dim);}

/* Icônes et couleurs */
.icon-const { color: #ffaa33; }
.icon-param { color: #33ccff; }
.icon-global { color: #ffaa33; }
.icon-local { color: #99cc99; }
.icon-unused { color: #ff6666; }
.icon-return { color: #ffaa33; }
.icon-dead { color: #ff6666; }

/* Style pour les résumés pliables dans les cartes */
.card details > summary {
    list-style: none;
    cursor: pointer;
    color: var(--blue);
    font-weight: 600;
}
.card details > summary::-webkit-details-marker {
    display: none;
}
.card details > summary::before {
    content: '▶';
    color: var(--dim);
    display: inline-block;
    width: 1.2em;
    font-size: 9px;
}
.card details[open] > summary::before {
    content: '▼';
}
</style>
"""

TRIVIAL_VALUES = {'0', '1', '0.0', '1.0', '0.5', 'True', 'False', 'None', "''", '""'}

def _esc(s: str) -> str:
    return (s.replace('&', '&amp;').replace('<', '&lt;')
             .replace('>', '&gt;').replace('"', '&quot;'))

def _build_dependency_graph(modules: List[ModuleInfo]) -> Dict[str, List[str]]:
    graph = {}
    for m in modules:
        if m.path.suffix == '.py':
            module_id = f"{m.layer}/{m.path.stem}"
            graph[module_id] = m.imports_prc
    return graph

def _render_tree(modules: List[ModuleInfo], root: Path) -> str:
    tree = {}
    for m in modules:
        try:
            rel = m.path.relative_to(root)
            parts = rel.parts
        except ValueError:
            continue
        node = tree
        for p in parts[:-1]:
            node = node.setdefault(p, {})
        node[parts[-1]] = None

    def _render_node(d, level=0):
        html = ''
        indent = level * 20
        for name, sub in sorted(d.items()):
            if sub is None:  # fichier
                ext = Path(name).suffix
                cls = 'file-py' if ext == '.py' else 'file-yaml'
                html += f'<div style="margin-left:{indent}px" class="{cls}">{_esc(name)}</div>\n'
            else:  # dossier
                html += f'<details style="margin-left:{indent}px">\n'
                html += f'<summary class="dir">{_esc(name)}/</summary>\n'
                html += _render_node(sub, level+1)
                html += f'</details>\n'
        return html

    return f'<div class="tree">\n{_render_node(tree, 0)}</div>'

def _render_var_list(vars_list: List[VariableDetail]) -> List[str]:
    items = []
    for v in vars_list:
        symbols = []
        if v.is_constant:
            symbols.append('<i class="fas fa-globe icon-const"></i>')
        elif v.is_global:
            symbols.append('<i class="fas fa-globe icon-global"></i>')
        if v.is_param:
            symbols.append('<i class="fas fa-cog icon-param"></i>')
        if not (v.is_global or v.is_param or v.is_constant) and not v.is_returned:
            symbols.append('<i class="fas fa-cube icon-local"></i>')
        if v.is_returned:
            symbols.append('<i class="fas fa-undo-alt icon-return"></i>')
        display = v.name
        if v.is_unused:
            span_class = 'icon-unused'
        elif v.is_global or v.is_param or v.is_returned or v.is_constant:
            span_class = 'icon-return'
        else:
            span_class = ''
        full = f'<span class="{span_class}">{" ".join(symbols)} {_esc(display)}</span>'
        items.append(full)
    return items

def _render_module_card(m: ModuleInfo,
                        constants: List[Dict],
                        unused_funcs: List[str],
                        unused_objs: List[UnusedIssue],
                        leak_objs: List[UnusedIssue],
                        unused_vars: List[UnusedIssue],
                        unused_imports: List[UnusedIssue],
                        is_dead: bool) -> str:
    role = _esc(m.doc.role or m.path.name)
    dead_flag = ' <span class="icon-dead"><i class="fas fa-fire"></i> module mort</span>' if is_dead else ''
    html = f'''<div class="card">
<div class="card-title">
  {_esc(m.path.name)}
  <span class="layer-badge">{_esc(m.layer)}</span>
  {dead_flag}
</div>
<div class="note" style="margin-bottom:6px">{role}</div>
'''

    # ----- Constantes (regroupées par valeur) -----
    const_map = {}
    for c in m.constants:
        key = c.value
        if key in TRIVIAL_VALUES:
            continue
        if key not in const_map:
            const_map[key] = {'names': [], 'lines': []}
        const_map[key]['names'].append(c.name)
        const_map[key]['lines'].append(c.line)
    for obs in constants:
        key = obs['value']
        if key in TRIVIAL_VALUES:
            continue
        if key not in const_map:
            const_map[key] = {'names': [], 'lines': []}
        const_map[key]['names'].append(obs['name'])
        const_map[key]['lines'].append(obs['line'])

    if const_map:
        const_display = []
        for val, info in const_map.items():
            if len(info['names']) == 1:
                name = info['names'][0]
                line = info['lines'][0]
                if name == '(literal)':
                    const_display.append(f'<i class="fas fa-pencil-alt"></i> {val} (l.{line})')
                else:
                    const_display.append(f'<i class="fas fa-globe icon-const"></i> {name}={val} (l.{line})')
            else:
                lines_str = ', '.join(str(l) for l in info['lines'][:3])
                suffix = f" (l.{lines_str}{'...' if len(info['lines'])>3 else ''})"
                if all(n == '(literal)' for n in info['names']):
                    const_display.append(f'<i class="fas fa-pencil-alt"></i> {val} apparaît {len(info["names"])} fois{suffix}')
                else:
                    first_name = info['names'][0]
                    const_display.append(f'<i class="fas fa-globe icon-const"></i> {first_name}={val} (et {len(info["names"])-1} autres){suffix}')
        html += f'<details style="margin-bottom:6px">'
        html += f'<summary style="color:var(--blue);"><i class="fas fa-chevron-right"></i> Constantes ({len(const_map)})</summary>'
        html += f'<div style="margin-top:4px; margin-left:10px;"><span class="note">{", ".join(_esc(s) for s in const_display)}</span></div>'
        html += f'</details>'

    # ----- Fonctions avec détails -----
    if m.functions:
        total_funcs = len(m.functions)
        dead_funcs_count = sum(1 for f in m.functions if f.name in unused_funcs)
        summary = f'Fonctions ({total_funcs}'
        if dead_funcs_count > 0:
            summary += f', <span style="color:var(--red)">{dead_funcs_count} morte{"s" if dead_funcs_count>1 else ""}</span>'
        summary += ')'
        html += f'<details style="margin-bottom:6px">'
        html += f'<summary style="color:var(--blue);">{summary}</summary>'
        html += '<div style="margin-top:4px; margin-left:10px;">'
        for func in sorted(m.functions, key=lambda f: f.name):
            is_dead_func = func.name in unused_funcs
            color = 'var(--red)' if is_dead_func else 'var(--green)'
            signature = func.signature
            if signature.startswith('def '):
                signature = signature[4:]

            html += f'<details class="function-details">'
            html += f'<summary style="color:{color};">'
            if is_dead_func:
                html += '<i class="fas fa-skull icon-dead"></i> '
            html += f'def {_esc(signature)}</summary>'
            html += '<div style="margin-left:10px; margin-top:4px;">'

            fd = m.function_details.get(func.name)
            if fd:
                if fd.calls_internal:
                    html += f'<span style="color:var(--dim)">  <strong style="color:var(--cyan)">appelle:</strong> {", ".join(_esc(c) for c in fd.calls_internal)}</span><br>'
                if fd.calls_external:
                    truncated = fd.calls_external[:5]
                    suffix = '...' if len(fd.calls_external) > 5 else ''
                    html += f'<span style="color:var(--dim)">  <strong style="color:var(--cyan)">appelle (ext):</strong> {", ".join(_esc(c) for c in truncated)}{suffix}</span><br>'

                if fd.reads:
                    items = _render_var_list(fd.reads)
                    html += f'<span style="color:var(--dim)">  <strong style="color:var(--cyan)">lit:</strong> {", ".join(items)}</span><br>'
                if fd.writes:
                    items = _render_var_list(fd.writes)
                    html += f'<span style="color:var(--dim)">  <strong style="color:var(--cyan)">écrit:</strong> {", ".join(items)}</span><br>'
                if fd.deletes:
                    items = _render_var_list(fd.deletes)
                    html += f'<span style="color:var(--dim)">  <strong style="color:var(--cyan)">supprime:</strong> {", ".join(items)}</span><br>'

            html += '</div></details>'
        html += '</div></details>'

    # ----- Fuites potentielles -----
    if leak_objs:
        leak_list = []
        for issue in leak_objs:
            parts = [f"{issue.name}"]
            if issue.extra and issue.extra.get('obj_type'):
                parts[0] += f" ({issue.extra['obj_type']})"
            if issue.line:
                parts.append(f"l.{issue.line}")
            if issue.extra and issue.extra.get('reason'):
                parts.append(f"⚠️ {issue.extra['reason']}")
            leak_list.append(' '.join(parts))
        html += f'<div style="margin-bottom:6px"><span style="color:var(--orange)"><strong style="color:var(--cyan)">Fuites potentielles (utilisé jamais supprimé):</strong></span> <span class="note">{", ".join(_esc(s) for s in leak_list)}</span></div>'

    # ----- Objets inutilisés -----
    if unused_objs:
        obj_list = []
        for issue in unused_objs:
            parts = [f"{issue.name} ({issue.extra.get('obj_type', '?')}) l.{issue.line}"]
            if issue.extra.get('is_global'):
                parts.insert(0, '<i class="fas fa-globe"></i>')
            obj_list.append(' '.join(parts))
        html += f'<div style="margin-bottom:6px"><span style="color:var(--orange)"><strong style="color:var(--cyan)">Objets inutilisés:</strong></span> <span class="note">{", ".join(_esc(s) for s in obj_list)}</span></div>'

    # ----- Variables inutilisées -----
    if unused_vars:
        var_list = []
        for issue in unused_vars:
            parts = [f"{issue.name} (l.{issue.line})"]
            if issue.function:
                parts.append(f"dans {issue.function}")
            var_list.append(' '.join(parts))
        html += f'<div style="margin-bottom:6px"><span style="color:var(--orange)"><strong style="color:var(--cyan)">Variables inutilisées:</strong></span> <span class="note">{", ".join(_esc(s) for s in var_list)}</span></div>'

    # ----- Imports inutilisés -----
    if unused_imports:
        imp_list = [f"{imp.name} (l.{imp.line})" for imp in unused_imports]
        html += f'<div style="margin-bottom:6px"><span style="color:var(--orange)"><strong style="color:var(--cyan)">Imports inutilisés:</strong></span> <span class="note">{", ".join(_esc(s) for s in imp_list)}</span></div>'

    # Métadonnées
    if m.metadata:
        html += '<div style="margin-bottom:6px"><span style="color:var(--purple)">Métadonnées:</span> '
        meta_str = ', '.join(f"{k}={v}" for k, v in m.metadata.items())
        html += f'<span class="note">{_esc(meta_str)}</span></div>'

    # Imports
    if m.imports_prc or m.imports_ext:
        html += '<div style="margin-top:6px">'
        if m.imports_prc:
            html += f'<span style="color:var(--blue)">Imports PRC:</span> {", ".join(_esc(i) for i in m.imports_prc)}<br>'
        if m.imports_ext:
            html += f'<span style="color:var(--green)">Imports externes:</span> {", ".join(_esc(i) for i in m.imports_ext)}'
        html += '</div>'

    html += '</div>\n'
    return html

def render_html(data: ProjectData) -> str:
    n_modules = len([m for m in data.modules if m.path.suffix == '.py'])
    n_yaml = len(data.yaml_files)

    n_dead_modules = len([i for i in data.unused_issues if i.kind == 'module'])
    n_dead_functions = len([i for i in data.unused_issues if i.kind == 'function'])
    n_dead_vars = len([i for i in data.unused_issues if i.kind in ('variable', 'object')])
    n_constants_unmatched = len(data.constant_observations)

    html = f'''<!DOCTYPE html>
<html lang="fr">
<head>
<meta charset="UTF-8">
<title>PRC — Architecture (v2 neutre)</title>
{_CSS}
</head>
<body>
<h1>PRC — Architecture Map (v2)</h1>
<div class="meta">
  <b>Généré</b> {_esc(data.generated_at)} &nbsp;·&nbsp;
  <b>Racine</b> {_esc(str(data.root))}
</div>

<div class="stats">
  <div class="stat"><div class="stat-n">{n_modules}</div><div class="stat-l"><i class="fas fa-file-code"></i> MODULES .py</div></div>
  <div class="stat"><div class="stat-n">{n_yaml}</div><div class="stat-l"><i class="fas fa-file-alt"></i> FICHIERS .yaml</div></div>
  <div class="stat {'stat-red' if n_dead_modules else ''}"><div class="stat-n">{n_dead_modules}</div><div class="stat-l"><i class="fas fa-fire"></i> MODULES MORTS</div></div>
  <div class="stat {'stat-red' if n_dead_functions else ''}"><div class="stat-n">{n_dead_functions}</div><div class="stat-l"><i class="fas fa-skull"></i> FONCTIONS MORTES</div></div>
  <div class="stat {'stat-red' if n_dead_vars else ''}"><div class="stat-n">{n_dead_vars}</div><div class="stat-l"><i class="fas fa-trash-alt"></i> VARIABLES/OBJETS INUTILISÉS</div></div>
  <div class="stat {'stat-amber' if n_constants_unmatched else ''}"><div class="stat-n">{n_constants_unmatched}</div><div class="stat-l"><i class="fas fa-exclamation-triangle"></i> CONSTANTES NON PARAMÉTRÉES</div></div>
</div>
'''

    # Arborescence
    html += '<div class="sl">Arborescence réelle</div>\n'
    html += '<div class="card">\n'
    html += _render_tree(data.modules, data.root)
    html += '</div>\n'

    # Préparation des données unused
    unused_funcs_by_module = defaultdict(list)
    unused_objs_by_module = defaultdict(list)
    unused_vars_by_module = defaultdict(list)
    unused_imports_by_module = defaultdict(list)
    leak_objs_by_module = defaultdict(list)
    dead_modules_set = set()

    for issue in data.unused_issues:
        if issue.kind == 'function':
            unused_funcs_by_module[issue.module].append(issue.name)
        elif issue.kind == 'object':
            unused_objs_by_module[issue.module].append(issue)
        elif issue.kind == 'variable':
            unused_vars_by_module[issue.module].append(issue)
        elif issue.kind == 'import':
            unused_imports_by_module[issue.module].append(issue)
        elif issue.kind == 'module':
            dead_modules_set.add(issue.module)
        elif issue.kind == 'potential_leak':
            leak_objs_by_module[issue.module].append(issue)

    # Constantes par module
    const_by_module = defaultdict(list)
    for obs in data.constant_observations:
        const_by_module[obs['module']].append(obs)

    # Regroupement par layer
    layers: Dict[str, List[ModuleInfo]] = defaultdict(list)
    for m in data.modules:
        if m.path.suffix == '.py':
            layers[m.layer].append(m)

    EXCLUDED_LAYERS = {'atomics'}
    for layer in sorted(layers.keys()):
        if layer in EXCLUDED_LAYERS:
            continue
        mods = layers[layer]
        html += f'<div class="sl">Layer : {_esc(layer)}</div>\n'
        html += '<div class="grid2">\n'
        for m in sorted(mods, key=lambda x: x.path.name):
            module_id = f"{m.layer}/{m.path.stem}"
            consts = const_by_module.get(module_id, [])
            unused_funcs = unused_funcs_by_module.get(module_id, [])
            unused_objs = unused_objs_by_module.get(module_id, [])
            unused_vars = unused_vars_by_module.get(module_id, [])
            unused_imports = unused_imports_by_module.get(module_id, [])
            leak_objs = leak_objs_by_module.get(module_id, [])
            is_dead = module_id in dead_modules_set
            html += _render_module_card(m, consts, unused_funcs, unused_objs, leak_objs,
                                        unused_vars, unused_imports, is_dead)
        html += '</div>\n'

    # Fichiers YAML
    if data.yaml_files:
        html += '<div class="sl">Fichiers YAML — inventaire clés top-level</div>\n'
        html += '<div class="grid3">\n'
        for yf in data.yaml_files:
            path_str = _esc(str(yf.path))
            keys_str = ', '.join(_esc(k) for k in yf.top_keys)
            desc = _esc(yf.description) if yf.description else ''
            html += f'<div class="card">'
            html += f'<div class="card-title">{Path(path_str).name}</div>'
            html += f'<div class="note">{path_str}</div>'
            if desc:
                html += f'<div class="note" style="margin-top:4px;color:var(--dim);font-style:italic;">{desc}</div>'
            html += f'<div class="note" style="margin-top:4px;color:var(--cyan)">{keys_str or "—"}</div>'
            html += f'</div>\n'
        html += '</div>\n'

    # Graphe des dépendances
    graph = _build_dependency_graph(data.modules)
    html += '<div class="sl">Dépendances entre modules</div>\n'
    html += '<div class="card">\n'
    if graph:
        for module, deps in sorted(graph.items()):
            if deps:
                html += f'<div><span style="color:var(--cyan)">{_esc(module)}</span> → {", ".join(_esc(d) for d in deps)}</div>\n'
    else:
        html += '<div class="note">Aucune dépendance détectée.</div>\n'
    html += '</div>\n'

    html += f'''<div style="color:var(--dim);font-size:10px;text-align:center;
padding:24px 0;border-top:1px solid var(--border);margin-top:32px;">
  Généré par generate_architecture_v2.py — {_esc(data.generated_at)}
</div>
</body></html>'''

    return html
