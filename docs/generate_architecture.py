"""
generate_architecture.py

Générateur automatique de architecture.html et requirements.txt vivant.

Usage :
    python generate_architecture.py [root_path]
    root_path : racine du projet PRC (défaut : répertoire courant)

Sortie :
    architecture.html  — carte complète du pipeline
    requirements.txt   — dépendances détectées depuis l'environnement actif

@ROLE    Générer architecture.html et requirements.txt depuis les sources PRC
@LAYER   root
@EXPORTS
  collect_all(root_path) → ArchitectureData  | orchestre sections 1-8
  main()                 → None              | entrypoint CLI
@LIFECYCLE
  CREATES  ArchitectureData  résultat collect_all, passé à render_html
  CREATES  html_output       str HTML, écrit sur disque puis libéré
  DELETES  html_output       après écriture fichier
@CONFORMITY
  OK — aucun import PRC (fichier standalone)
@BUGS
@TODOS
@QUESTIONS
"""

from __future__ import annotations

import ast
import os
import sys
import textwrap
from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 1 — SCANNER
# ─────────────────────────────────────────────────────────────────────────────

EXCLUDE_DIRS  = {'tests', 'docs', 'data', '__pycache__', '.git', '.ipynb_checkpoints'}
VALID_EXTS    = {'.py', '.yaml'}


def scan_files(root_path: str) -> List[Path]:
    """
    Walk récursif depuis root_path.
    Exclut : dossiers __*, tests/, docs/, data/, __pycache__, .git
    Fichiers valides : .py et .yaml uniquement.
    Retourne chemins absolus triés.
    """
    root = Path(root_path).resolve()
    results: List[Path] = []

    for dirpath, dirnames, filenames in os.walk(root):
        current = Path(dirpath)

        # Élagage des dossiers exclus (in-place pour os.walk)
        dirnames[:] = [
            d for d in dirnames
            if d not in EXCLUDE_DIRS and not d.startswith('__')
        ]

        for fname in filenames:
            fpath = current / fname
            if fpath.suffix in VALID_EXTS and not fname.startswith('__'):
                results.append(fpath)

    return sorted(results)


def _detect_prc_layers(files: List[Path], root: Path) -> set:
    """
    Infère les layers PRC depuis l'arborescence.
    Un layer = dossier immédiat sous root contenant au moins un .py.
    """
    layers = set()
    for f in files:
        if f.suffix == '.py':
            try:
                rel = f.relative_to(root)
                if len(rel.parts) > 1:
                    layers.add(rel.parts[0])
            except ValueError:
                pass
    return layers


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 2 — AST PARSER
# ─────────────────────────────────────────────────────────────────────────────

_STDLIB = sys.stdlib_module_names if hasattr(sys, 'stdlib_module_names') else set()


def _top_package(module_str: Optional[str]) -> str:
    if not module_str:
        return ''
    return module_str.split('.')[0]


def parse_imports(filepath: Path, prc_layers: set) -> Tuple[List[str], List[str]]:
    """
    AST parsing d'un fichier .py → (imports_prc, imports_externes).
    imports_prc      : modules appartenant aux layers PRC
    imports_externes : packages tiers (ni stdlib, ni PRC)
    """
    try:
        source = filepath.read_text(encoding='utf-8', errors='replace')
        tree   = ast.parse(source, filename=str(filepath))
    except SyntaxError:
        return [], []

    prc_imports: List[str] = []
    ext_imports: List[str] = []

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                pkg = _top_package(alias.name)
                _classify(pkg, prc_layers, prc_imports, ext_imports)
        elif isinstance(node, ast.ImportFrom):
            pkg = _top_package(node.module)
            if pkg:
                _classify(pkg, prc_layers, prc_imports, ext_imports)

    return sorted(set(prc_imports)), sorted(set(ext_imports))


def _classify(pkg: str, prc_layers: set,
              prc_imports: List[str], ext_imports: List[str]) -> None:
    if not pkg:
        return
    if pkg in prc_layers:
        prc_imports.append(pkg)
    elif pkg not in _STDLIB and pkg not in ('', '__future__'):
        ext_imports.append(pkg)


def resolve_versions(imports_externes: List[str]) -> Dict[str, str]:
    """
    Résout les versions depuis l'environnement actif via importlib.metadata.
    Retourne {package: version | 'unknown'}.
    """
    # Mapping import name → PyPI package name
    _IMPORT_TO_PYPI = {
        'yaml': 'PyYAML',
        'cv2': 'opencv-python',
        'PIL': 'Pillow',
        'sklearn': 'scikit-learn',
        'bs4': 'beautifulsoup4',
        'jax': 'jax',
        'umap': 'umap-learn',
    }
    import importlib.metadata as meta
    result: Dict[str, str] = {}
    for pkg in imports_externes:
        pypi_name = _IMPORT_TO_PYPI.get(pkg, pkg)
        try:
            result[pkg] = meta.version(pypi_name)
        except meta.PackageNotFoundError:
            try:
                result[pkg] = meta.version(pkg.replace('_', '-'))
            except meta.PackageNotFoundError:
                result[pkg] = 'unknown'
    return result


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 3 — DOCSTRING PARSER
# ─────────────────────────────────────────────────────────────────────────────

KNOWN_SECTIONS = {
    '@ROLE', '@LAYER', '@EXPORTS', '@LIFECYCLE',
    '@CONFORMITY', '@BUGS', '@TODOS', '@QUESTIONS',
}


def _parse_section_lines(lines: List[str], start: int) -> List[str]:
    """
    Extrait les lignes indentées sous un @MARQUEUR jusqu'au prochain @MARQUEUR.
    Ligne inline (même ligne que le marqueur) incluse.
    """
    result: List[str] = []
    i = start + 1
    while i < len(lines):
        line = lines[i]
        stripped = line.strip()
        if stripped.startswith('@') and stripped.split()[0].upper() in KNOWN_SECTIONS:
            break
        if stripped:
            result.append(stripped)
        i += 1
    return result


def _parse_exports(lines: List[str]) -> List[Dict]:
    exports = []
    for line in lines:
        if '→' in line or '->' in line:
            sep   = '→' if '→' in line else '->'
            parts = line.split(sep, 1)
            sig   = parts[0].strip()
            rest  = parts[1].strip() if len(parts) > 1 else ''
            note_parts = rest.split('|', 1)
            ret  = note_parts[0].strip()
            note = note_parts[1].strip() if len(note_parts) > 1 else ''
            exports.append({'sig': sig, 'ret': ret, 'note': note})
        elif line.strip():
            exports.append({'sig': line.strip(), 'ret': '', 'note': ''})
    return exports


def _parse_lifecycle(lines: List[str]) -> List[Dict]:
    entries = []
    for line in lines:
        parts = line.split(None, 2)
        if len(parts) >= 2 and parts[0].upper() in ('CREATES', 'RECEIVES', 'PASSES', 'DELETES'):
            entries.append({
                'verb':    parts[0].upper(),
                'object':  parts[1],
                'context': parts[2].strip() if len(parts) > 2 else '',
            })
    return entries


def _parse_conformity(lines: List[str]) -> List[Dict]:
    items = []
    for line in lines:
        parts = line.split('—', 1)
        if not parts:
            continue
        status = parts[0].strip().upper()
        if status not in ('OK', 'WARN', 'VIOLATION'):
            status = 'WARN'
        rest = parts[1].strip() if len(parts) > 1 else line.strip()
        rule_note = rest.split('—', 1)
        items.append({
            'status': status,
            'rule':   rule_note[0].strip(),
            'note':   rule_note[1].strip() if len(rule_note) > 1 else '',
        })
    return items


def _parse_bugs(lines: List[str]) -> List[Dict]:
    bugs = []
    for line in lines:
        if '—' in line or '-' in line:
            sep   = '—' if '—' in line else ' - '
            parts = line.split(sep, 1)
            bugs.append({
                'id':   parts[0].strip(),
                'desc': parts[1].strip() if len(parts) > 1 else '',
            })
    return bugs


def _parse_todos(lines: List[str]) -> List[Dict]:
    todos = []
    for line in lines:
        blocks = ''
        l = line
        if '[blocks:' in l:
            b_start = l.index('[blocks:')
            b_end   = l.index(']', b_start)
            blocks  = l[b_start+8:b_end].strip()
            l       = l[:b_start] + l[b_end+1:]
        if '—' in l or ' - ' in l:
            sep   = '—' if '—' in l else ' - '
            parts = l.split(sep, 1)
            todos.append({
                'id':     parts[0].strip(),
                'blocks': blocks,
                'desc':   parts[1].strip() if len(parts) > 1 else '',
            })
        elif l.strip():
            todos.append({'id': '', 'blocks': blocks, 'desc': l.strip()})
    return todos


def _parse_questions(lines: List[str]) -> List[Dict]:
    qs = []
    for line in lines:
        if '—' in line or ' - ' in line:
            sep   = '—' if '—' in line else ' - '
            parts = line.split(sep, 1)
            qs.append({
                'id':   parts[0].strip(),
                'desc': parts[1].strip() if len(parts) > 1 else '',
            })
    return qs


def parse_docstring(filepath: Path) -> Dict:
    """
    Extrait et parse la docstring module d'un fichier .py.
    Retourne un dict avec toutes les sections — champs absents = valeur vide.
    """
    empty = {
        'role': '', 'layer': '',
        'exports': [], 'lifecycle': [],
        'conformity': [], 'bugs': [],
        'todos': [], 'questions': [],
    }
    try:
        source = filepath.read_text(encoding='utf-8', errors='replace')
        tree   = ast.parse(source)
    except SyntaxError:
        return empty

    raw_doc = ast.get_docstring(tree)
    if not raw_doc:
        return empty

    lines = raw_doc.splitlines()
    sections: Dict[str, List[str]] = {}
    current_section: Optional[str] = None

    for i, line in enumerate(lines):
        stripped = line.strip()
        token    = stripped.split()[0].upper() if stripped.split() else ''

        if token in KNOWN_SECTIONS:
            current_section = token
            # Valeur inline sur la même ligne
            rest = stripped[len(token):].strip()
            sections[current_section] = [rest] if rest else []
            # Lignes suivantes jusqu'au prochain marqueur
            sections[current_section] += _parse_section_lines(lines, i)
        # Lignes hors section ignorées (free text)

    result = dict(empty)
    result['role']       = ' '.join(sections.get('@ROLE', [])).strip()
    result['layer']      = ' '.join(sections.get('@LAYER', [])).strip()
    result['exports']    = _parse_exports(sections.get('@EXPORTS', []))
    result['lifecycle']  = _parse_lifecycle(sections.get('@LIFECYCLE', []))
    result['conformity'] = _parse_conformity(sections.get('@CONFORMITY', []))
    result['bugs']       = _parse_bugs(sections.get('@BUGS', []))
    result['todos']      = _parse_todos(sections.get('@TODOS', []))
    result['questions']  = _parse_questions(sections.get('@QUESTIONS', []))
    return result


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 4 — YAML PARSER
# ─────────────────────────────────────────────────────────────────────────────

def parse_yaml_meta(filepath: Path) -> Dict:
    """
    Charge un fichier YAML et extrait les clés top-level.
    Pas d'interprétation du contenu — inventaire uniquement.
    Retourne {path, keys: List[str]}.
    """
    try:
        import yaml
        with open(filepath, encoding='utf-8') as f:
            data = yaml.safe_load(f)
        keys = list(data.keys()) if isinstance(data, dict) else []
    except Exception as e:
        keys = [f'PARSE_ERROR: {e}']
    return {'path': filepath, 'keys': keys}


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 5 — GRAPH BUILDER + DATACLASSES
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class ModuleInfo:
    path        : Path
    layer       : str          # layer physique depuis arborescence
    imports_prc : List[str]
    imports_ext : List[str]
    doc         : Dict


@dataclass
class Violation:
    from_module : str
    to_module   : str
    rule        : str


# Règles conformité charter sur le graphe d'imports
# (from_layer, to_layer) → règle violée
_FORBIDDEN_IMPORTS: List[Tuple[str, str, str]] = [
    ('featuring', 'analysing', 'featuring → analysing interdit (P1)'),
    ('featuring', 'profiling', 'featuring → profiling interdit (P1)'),
    ('core',      'atomics',   'core aveugle : aucune connaissance atomics (K4)'),
    ('core',      'featuring', 'core aveugle : aucune connaissance featuring (K4)'),
    ('core',      'running',   'core aveugle : aucune connaissance running (K4)'),
    ('analysing', 'running',   'analysing → running interdit (sens du flot)'),
    ('profiling', 'running',   'profiling → running interdit (sens du flot)'),
]

# Layers qui ne doivent pas s'importer entre pairs (modules du même layer)
_NO_PEER_IMPORTS = {'featuring', 'analysing', 'profiling', 'running'}


def _get_layer(filepath: Path, root: Path) -> str:
    try:
        rel = filepath.relative_to(root)
        return rel.parts[0] if len(rel.parts) > 1 else 'root'
    except ValueError:
        return 'root'


def build_dependency_graph(
    modules : List[ModuleInfo],
    root    : Path,
) -> Tuple[Dict[str, List[str]], List[Violation]]:
    """
    Construit {module_name → [modules PRC importés]}.
    Détecte violations charter sur le sens du flot et isolation layers.
    Retourne (graph, violations).
    """
    graph: Dict[str, List[str]] = {}
    violations: List[Violation] = []

    # Index name → ModuleInfo
    name_map: Dict[str, ModuleInfo] = {}
    for m in modules:
        if m.path.suffix == '.py':
            name = m.path.stem
            name_map[name] = m
            # Clé par layer/module pour imports PRC
            layer_key = f"{m.layer}/{m.path.stem}"
            graph[layer_key] = m.imports_prc

    for m in modules:
        if m.path.suffix != '.py':
            continue
        from_layer = m.layer
        for imported_layer in m.imports_prc:
            # Violation : imports interdits
            for (fl, tl, rule) in _FORBIDDEN_IMPORTS:
                if from_layer == fl and imported_layer == tl:
                    violations.append(Violation(
                        from_module=str(m.path.name),
                        to_module=imported_layer,
                        rule=rule,
                    ))
            # Violation : imports pair (même layer) — hubs exclus
            # Charter : hubs peuvent importer leurs modules de layer
            # Seuls les non-hubs ne peuvent pas s'appeler entre pairs
            is_hub = m.path.stem.startswith('hub_')
            if (from_layer == imported_layer
                    and from_layer in _NO_PEER_IMPORTS
                    and not is_hub):
                violations.append(Violation(
                    from_module=str(m.path.name),
                    to_module=imported_layer,
                    rule=f'import pair interdit dans layer {from_layer} (non-hub)',
                ))

    return graph, violations


def topological_sort(
    modules : List[ModuleInfo],
    root    : Path,
) -> List[str]:
    """
    Tri topologique BFS depuis batch.py (ou root si absent).
    Retourne liste ordonnée de 'layer/module'.
    Cycles → flag dans la liste ('CYCLE DÉTECTÉ').
    """
    # Construit graphe layer-level
    layer_graph: Dict[str, set] = defaultdict(set)
    layers_seen: set = set()

    for m in modules:
        if m.path.suffix != '.py':
            continue
        layer = m.layer
        layers_seen.add(layer)
        for imp in m.imports_prc:
            if imp != layer:
                layer_graph[layer].add(imp)

    # BFS depuis 'root' (batch.py)
    visited = []
    queue   = deque(['root'])
    seen    = set()

    while queue:
        node = queue.popleft()
        if node in seen:
            continue
        seen.add(node)
        visited.append(node)
        for dep in sorted(layer_graph.get(node, [])):
            if dep not in seen:
                queue.append(dep)

    # Ajouter layers non atteints
    for layer in sorted(layers_seen):
        if layer not in seen:
            visited.append(layer)

    return visited


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 6 — LIFECYCLE ANALYZER
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class LeakFlag:
    object_name : str
    created_in  : str
    context     : str


@dataclass
class MismatchFlag:
    object_name : str
    passed_from : str
    context     : str


@dataclass
class LifecycleReport:
    leaks      : List[LeakFlag]      = field(default_factory=list)
    mismatches : List[MismatchFlag]  = field(default_factory=list)
    clean      : List[str]           = field(default_factory=list)  # objets avec cycle complet


def analyze_lifecycle(modules: List[ModuleInfo]) -> LifecycleReport:
    """
    Agrège CREATES/PASSES/RECEIVES/DELETES de tous les modules.
    CREATES sans DELETES nulle part → LeakFlag.
    PASSES sans RECEIVES correspondant → MismatchFlag.
    """
    creates:  Dict[str, List[Tuple[str, str]]] = defaultdict(list)  # obj → [(module, ctx)]
    deletes:  set = set()
    passes:   Dict[str, List[Tuple[str, str]]] = defaultdict(list)
    receives: set = set()

    for m in modules:
        if m.path.suffix != '.py':
            continue
        mod_name = f"{m.layer}/{m.path.stem}"
        for lc in m.doc.get('lifecycle', []):
            verb = lc['verb']
            obj  = lc['object']
            ctx  = lc['context']
            if verb == 'CREATES':
                creates[obj].append((mod_name, ctx))
            elif verb == 'DELETES':
                deletes.add(obj)
            elif verb == 'PASSES':
                passes[obj].append((mod_name, ctx))
            elif verb == 'RECEIVES':
                receives.add(obj)

    report = LifecycleReport()

    for obj, locs in creates.items():
        if obj in deletes:
            report.clean.append(obj)
        else:
            for (mod, ctx) in locs:
                report.leaks.append(LeakFlag(
                    object_name=obj,
                    created_in=mod,
                    context=ctx,
                ))

    for obj, locs in passes.items():
        if obj not in receives:
            for (mod, ctx) in locs:
                report.mismatches.append(MismatchFlag(
                    object_name=obj,
                    passed_from=mod,
                    context=ctx,
                ))

    return report


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 7 — CONFORMITY CHECKER
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class ConformityItem:
    module : str
    status : str   # OK | WARN | VIOLATION
    source : str   # 'calculated' | 'declared'
    rule   : str
    note   : str = ''


def check_conformity(
    modules           : List[ModuleInfo],
    graph_violations  : List[Violation],
) -> List[ConformityItem]:
    """
    Fusionne violations calculées (depuis graph) + déclarées (@CONFORMITY).
    Signale les discordances (déclaré OK mais calculé VIOLATION).
    """
    items: List[ConformityItem] = []

    # Violations calculées → ConformityItem
    calc_by_module: Dict[str, List[Violation]] = defaultdict(list)
    for v in graph_violations:
        calc_by_module[v.from_module].append(v)

    for m in modules:
        if m.path.suffix != '.py':
            continue
        mod_name = m.path.name

        # Violations calculées
        for v in calc_by_module.get(mod_name, []):
            items.append(ConformityItem(
                module=f"{m.layer}/{mod_name}",
                status='VIOLATION',
                source='calculated',
                rule=v.rule,
            ))

        # Déclarées dans @CONFORMITY
        for c in m.doc.get('conformity', []):
            items.append(ConformityItem(
                module=f"{m.layer}/{mod_name}",
                status=c['status'],
                source='declared',
                rule=c['rule'],
                note=c['note'],
            ))

        # Discordance : module déclaré OK mais violation calculée dessus
        calc_violations = {v.rule for v in calc_by_module.get(mod_name, [])}
        declared_oks    = {c['rule'] for c in m.doc.get('conformity', []) if c['status'] == 'OK'}
        for rule in calc_violations & declared_oks:
            items.append(ConformityItem(
                module=f"{m.layer}/{mod_name}",
                status='VIOLATION',
                source='calculated',
                rule=f'DISCORDANCE — déclaré OK mais calculé VIOLATION : {rule}',
            ))

    return items


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 8 — REQUIREMENTS GENERATOR
# ─────────────────────────────────────────────────────────────────────────────

def generate_requirements(modules: List[ModuleInfo]) -> str:
    """
    Union de tous les imports externes détectés.
    Filtre les layers PRC (filet de sécurité si prc_layers incomplet au parse).
    Versions résolues depuis l'environnement actif.
    Format : 'package>=version' trié alphabétiquement.
    """
    # Layers PRC inférés depuis les ModuleInfo — filtre de sécurité
    prc_layers = {m.layer for m in modules if m.layer and m.layer not in ('root', '')}

    all_ext: set = set()
    for m in modules:
        for pkg in m.imports_ext:
            if pkg not in prc_layers:
                all_ext.add(pkg)

    versions = resolve_versions(sorted(all_ext))

    lines = [
        f"# requirements.txt — PRC Pipeline",
        f"# Généré automatiquement par generate_architecture.py",
        f"# {datetime.now().strftime('%Y-%m-%d %H:%M')}",
        f"# Ne pas éditer manuellement — éditer les sources .py",
        "",
    ]

    for pkg in sorted(versions):
        v = versions[pkg]
        if v and v != 'unknown':
            lines.append(f"{pkg}>={v}")
        else:
            lines.append(f"{pkg}  # version inconnue dans l'environnement actif")

    return '\n'.join(lines)


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 9 — HTML RENDERER
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class ArchitectureData:
    modules      : List[ModuleInfo]
    topo_order   : List[str]
    graph        : Dict[str, List[str]]
    lifecycle    : LifecycleReport
    conformity   : List[ConformityItem]
    yaml_files   : List[Dict]
    generated_at : str
    root         : Path


_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@300;400;600;700&family=Syne:wght@400;600;800&display=swap');
:root{
  --bg:#080a0d;--bg2:#0d1117;--bg3:#131920;--border:#1a2030;--border2:#243040;
  --text:#b0bcc8;--dim:#485060;--bright:#dce6f0;
  --blue:#4a9eff;--cyan:#00d4c8;--green:#3dba6a;--amber:#f0a020;
  --red:#e05050;--purple:#9a70f0;--slate:#5a7090;--orange:#e07830;
  --mono:'JetBrains Mono',monospace;
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
/* flags */
.flag{font-size:9px;padding:1px 7px;border-radius:2px;
      display:inline-block;margin:2px 2px 2px 0;font-weight:600;}
.f-bug    {background:rgba(224,80,80,.15);color:var(--red);border:1px solid rgba(224,80,80,.3);}
.f-todo   {background:rgba(224,120,48,.15);color:var(--orange);border:1px solid rgba(224,120,48,.3);}
.f-q      {background:rgba(90,112,144,.15);color:var(--slate);border:1px solid rgba(90,112,144,.3);}
.f-ok     {background:rgba(61,186,106,.10);color:var(--green);border:1px solid rgba(61,186,106,.25);}
.f-warn   {background:rgba(240,160,32,.12);color:var(--amber);border:1px solid rgba(240,160,32,.25);}
.f-viol   {background:rgba(224,80,80,.12);color:var(--red);border:1px solid rgba(224,80,80,.25);}
.f-calc   {background:rgba(154,112,240,.12);color:var(--purple);border:1px solid rgba(154,112,240,.25);}
.f-leak   {background:rgba(224,80,80,.20);color:var(--red);border:1px solid rgba(224,80,80,.4);}
.f-clean  {background:rgba(61,186,106,.12);color:var(--green);border:1px solid rgba(61,186,106,.3);}
.f-mis    {background:rgba(240,160,32,.18);color:var(--amber);border:1px solid rgba(240,160,32,.35);}
/* tree */
.tree{font-size:10px;line-height:1.9;color:var(--text);}
.tree .dir{color:var(--blue);}
.tree .file-py{color:var(--cyan);}
.tree .file-yaml{color:var(--amber);}
/* flow */
.flow{display:flex;flex-wrap:wrap;align-items:center;gap:6px;margin:6px 0;}
.flow-node{font-size:10px;padding:3px 10px;background:var(--bg3);
           border:1px solid var(--border2);color:var(--bright);}
.flow-arrow{color:var(--dim);font-size:12px;}
/* table lifecycle */
table{width:100%;border-collapse:collapse;font-size:10px;margin-top:6px;}
th{color:var(--dim);font-weight:600;text-align:left;padding:4px 8px;
   border-bottom:1px solid var(--border2);letter-spacing:.06em;font-size:9px;}
td{padding:4px 8px;border-bottom:1px solid var(--border);vertical-align:top;}
td.obj{color:var(--cyan);}
td.mod{color:var(--blue);}
td.ctx{color:var(--dim);}
td.st-leak{color:var(--red);}
td.st-clean{color:var(--green);}
td.st-mis{color:var(--amber);}
.stats{display:flex;gap:24px;margin-bottom:20px;flex-wrap:wrap;}
.stat{background:var(--bg2);border:1px solid var(--border);
      padding:10px 16px;min-width:100px;}
.stat-n{font-size:20px;font-weight:700;color:var(--bright);}
.stat-l{font-size:9px;color:var(--dim);margin-top:2px;letter-spacing:.06em;}
.stat-red .stat-n{color:var(--red);}
.stat-amber .stat-n{color:var(--amber);}
.stat-green .stat-n{color:var(--green);}
</style>
"""


def _esc(s: str) -> str:
    return (s.replace('&', '&amp;').replace('<', '&lt;')
             .replace('>', '&gt;').replace('"', '&quot;'))


def _render_tree(modules: List[ModuleInfo], root: Path) -> str:
    """Arborescence physique reconstruite."""
    tree: Dict = {}
    for m in modules:
        try:
            rel  = m.path.relative_to(root)
            parts = rel.parts
        except ValueError:
            continue
        node = tree
        for p in parts[:-1]:
            node = node.setdefault(p, {})
        node[parts[-1]] = None

    def _render_node(d: Dict, indent: int) -> str:
        html = ''
        prefix = '&nbsp;' * (indent * 4)
        for k in sorted(d.keys()):
            v = d[k]
            if v is None:
                ext = Path(k).suffix
                cls = 'file-py' if ext == '.py' else 'file-yaml'
                html += f'<div class="{cls}">{prefix}{_esc(k)}</div>\n'
            else:
                html += f'<div class="dir">{prefix}{_esc(k)}/</div>\n'
                html += _render_node(v, indent + 1)
        return html

    return f'<div class="tree">\n{_render_node(tree, 0)}</div>'


def _render_pipeline_flow(topo_order: List[str]) -> str:
    html = '<div class="flow">'
    for i, layer in enumerate(topo_order):
        html += f'<span class="flow-node">{_esc(layer)}</span>'
        if i < len(topo_order) - 1:
            html += '<span class="flow-arrow">→</span>'
    html += '</div>'
    return html


def _render_module_card(
    m             : ModuleInfo,
    conf_items    : List[ConformityItem],
    lifecycle     : LifecycleReport,
) -> str:
    mod_id = f"{m.layer}/{m.path.stem}"

    # Header
    role = _esc(m.doc.get('role', '') or m.path.name)
    html = f'''<div class="card">
<div class="card-title">
  {_esc(m.path.name)}
  <span class="layer-badge">{_esc(m.layer)}</span>
</div>
<div class="note" style="margin-bottom:6px">{role}</div>
'''

    # Exports
    exports = m.doc.get('exports', [])
    if exports:
        html += '<div style="margin-bottom:6px">'
        for e in exports:
            sig  = _esc(e.get('sig', ''))
            ret  = _esc(e.get('ret', ''))
            note = _esc(e.get('note', ''))
            ret_part  = f' → <span style="color:var(--green)">{ret}</span>' if ret else ''
            note_part = f' <span style="color:var(--dim)">| {note}</span>' if note else ''
            html += f'<div class="fn">{sig}{ret_part}{note_part}</div>\n'
        html += '</div>'

    # Lifecycle de ce module
    lc_entries = m.doc.get('lifecycle', [])
    if lc_entries:
        html += '<div style="margin-top:6px;margin-bottom:4px">'
        for lc in lc_entries:
            verb = lc['verb']
            obj  = _esc(lc['object'])
            ctx  = _esc(lc['context'])
            color = {
                'CREATES':  'var(--cyan)',
                'RECEIVES': 'var(--blue)',
                'PASSES':   'var(--amber)',
                'DELETES':  'var(--dim)',
            }.get(verb, 'var(--text)')
            html += (f'<div class="note">'
                     f'<span style="color:{color};font-weight:600">{verb}</span> '
                     f'<span style="color:var(--cyan)">{obj}</span>'
                     f'{" — " + ctx if ctx else ""}</div>\n')
        html += '</div>'

    # Fuites lifecycle concernant ce module
    leaks_here = [lk for lk in lifecycle.leaks if lk.created_in == mod_id]
    for lk in leaks_here:
        html += (f'<span class="flag f-leak">⚠ FUITE — '
                 f'{_esc(lk.object_name)}</span>\n')

    # Conformity
    if conf_items:
        html += '<div style="margin-top:6px">'
        for c in conf_items:
            cls   = {'OK': 'f-ok', 'WARN': 'f-warn', 'VIOLATION': 'f-viol'}.get(c.status, 'f-warn')
            src   = '⚙' if c.source == 'calculated' else '✎'
            label = f'{src} {c.status}'
            rule  = _esc(c.rule)
            note  = f' — {_esc(c.note)}' if c.note else ''
            html += f'<div><span class="flag {cls}">{label}</span> <span class="note">{rule}{note}</span></div>\n'
        html += '</div>'

    # TODOS + BUGS (todos orange, en premier, avec lien vers bug bloqué)
    todos = m.doc.get('todos', [])
    bugs  = m.doc.get('bugs', [])
    blocked_bugs = {t.get('blocks', '') for t in todos if t.get('blocks')}

    if todos:
        html += '<div style="margin-top:6px">'
        for t in todos:
            tid    = _esc(t.get('id', ''))
            desc   = _esc(t.get('desc', ''))
            blocks = t.get('blocks', '')
            b_part = f' <span style="color:var(--dim)">[blocks:{_esc(blocks)}]</span>' if blocks else ''
            html += (f'<div><span class="flag f-todo">TODO {tid}</span>'
                     f'{b_part} <span class="note">{desc}</span></div>\n')
        html += '</div>'

    if bugs:
        html += '<div style="margin-top:4px">'
        for b in bugs:
            bid  = _esc(b.get('id', ''))
            desc = _esc(b.get('desc', ''))
            # Grisé si un TODO le bloque
            dimmed = ' style="opacity:.5"' if bid in blocked_bugs else ''
            html += (f'<div{dimmed}><span class="flag f-bug">BUG {bid}</span>'
                     f' <span class="note">{desc}</span></div>\n')
        html += '</div>'

    # Questions
    questions = m.doc.get('questions', [])
    if questions:
        html += '<div style="margin-top:4px">'
        for q in questions:
            qid  = _esc(q.get('id', ''))
            desc = _esc(q.get('desc', ''))
            html += (f'<div><span class="flag f-q">? {qid}</span>'
                     f' <span class="note">{desc}</span></div>\n')
        html += '</div>'

    html += '</div>\n'
    return html


def _render_lifecycle_global(report: LifecycleReport) -> str:
    html = '<table>\n'
    html += ('<tr><th>OBJET</th><th>CRÉÉ DANS</th><th>CONTEXTE</th>'
             '<th>STATUT</th></tr>\n')

    for lk in report.leaks:
        html += (f'<tr><td class="obj">{_esc(lk.object_name)}</td>'
                 f'<td class="mod">{_esc(lk.created_in)}</td>'
                 f'<td class="ctx">{_esc(lk.context)}</td>'
                 f'<td class="st-leak">⚠ FUITE</td></tr>\n')

    for obj in sorted(report.clean):
        html += (f'<tr><td class="obj">{_esc(obj)}</td>'
                 f'<td class="mod">—</td><td class="ctx">—</td>'
                 f'<td class="st-clean">✓ clean</td></tr>\n')

    for ms in report.mismatches:
        html += (f'<tr><td class="obj">{_esc(ms.object_name)}</td>'
                 f'<td class="mod">{_esc(ms.passed_from)}</td>'
                 f'<td class="ctx">{_esc(ms.context)}</td>'
                 f'<td class="st-mis">⚠ PASSES sans RECEIVES</td></tr>\n')

    html += '</table>\n'
    return html


def render_html(data: ArchitectureData) -> str:
    """Construit le HTML complet depuis ArchitectureData."""

    # Stats globales
    n_modules    = sum(1 for m in data.modules if m.path.suffix == '.py')
    n_yaml       = sum(1 for m in data.modules if m.path.suffix == '.yaml')
    n_violations = sum(1 for c in data.conformity if c.status == 'VIOLATION')
    n_leaks      = len(data.lifecycle.leaks)
    n_bugs       = sum(len(m.doc.get('bugs', [])) for m in data.modules)
    n_todos      = sum(len(m.doc.get('todos', [])) for m in data.modules)
    n_questions  = sum(len(m.doc.get('questions', [])) for m in data.modules)

    html = f'''<!DOCTYPE html>
<html lang="fr">
<head>
<meta charset="UTF-8">
<title>PRC — Architecture</title>
{_CSS}
</head>
<body>
<h1>PRC — Architecture Map</h1>
<div class="meta">
  <b>Généré</b> {_esc(data.generated_at)} &nbsp;·&nbsp;
  <b>Racine</b> {_esc(str(data.root))}
</div>

<div class="stats">
  <div class="stat"><div class="stat-n">{n_modules}</div><div class="stat-l">MODULES .py</div></div>
  <div class="stat"><div class="stat-n">{n_yaml}</div><div class="stat-l">FICHIERS .yaml</div></div>
  <div class="stat {'stat-red' if n_violations else 'stat-green'}">
    <div class="stat-n">{n_violations}</div><div class="stat-l">VIOLATIONS</div></div>
  <div class="stat {'stat-red' if n_leaks else 'stat-green'}">
    <div class="stat-n">{n_leaks}</div><div class="stat-l">FUITES LIFECYCLE</div></div>
  <div class="stat {'stat-red' if n_bugs else 'stat-green'}">
    <div class="stat-n">{n_bugs}</div><div class="stat-l">BUGS OUVERTS</div></div>
  <div class="stat {'stat-amber' if n_todos else 'stat-green'}">
    <div class="stat-n">{n_todos}</div><div class="stat-l">TODOS</div></div>
  <div class="stat">
    <div class="stat-n">{n_questions}</div><div class="stat-l">QUESTIONS</div></div>
</div>
'''

    # Arborescence
    html += '<div class="sl">Arborescence réelle</div>\n'
    html += '<div class="card">\n'
    html += _render_tree(data.modules, data.root)
    html += '</div>\n'

    # Flux pipeline
    html += '<div class="sl">Flux pipeline — ordre topologique</div>\n'
    html += '<div class="card">\n'
    html += _render_pipeline_flow(data.topo_order)
    html += '</div>\n'

    # Lifecycle global
    html += '<div class="sl">Lifecycle global — RAM/VRAM</div>\n'
    html += '<div class="card">\n'
    html += _render_lifecycle_global(data.lifecycle)
    html += '</div>\n'

    # Conformité globale
    html += '<div class="sl">Conformité charter</div>\n'
    html += '<div class="card">\n'
    if data.conformity:
        for c in data.conformity:
            cls   = {'OK': 'f-ok', 'WARN': 'f-warn', 'VIOLATION': 'f-viol'}.get(c.status, 'f-warn')
            src   = '⚙ calculé' if c.source == 'calculated' else '✎ déclaré'
            html += (f'<div style="margin-bottom:4px">'
                     f'<span class="flag {cls}">{_esc(c.status)}</span> '
                     f'<span class="flag f-calc">{src}</span> '
                     f'<span style="color:var(--blue)">{_esc(c.module)}</span> — '
                     f'<span class="note">{_esc(c.rule)}'
                     f'{" — " + _esc(c.note) if c.note else ""}</span></div>\n')
    else:
        html += '<div class="note">Aucune violation détectée.</div>\n'
    html += '</div>\n'

    # Modules par layer
    layers: Dict[str, List[ModuleInfo]] = defaultdict(list)
    for m in data.modules:
        if m.path.suffix == '.py':
            layers[m.layer].append(m)

    conf_by_module: Dict[str, List[ConformityItem]] = defaultdict(list)
    for c in data.conformity:
        conf_by_module[c.module].append(c)

    for layer in data.topo_order:
        mods = layers.get(layer, [])
        if not mods:
            continue
        html += f'<div class="sl">Layer : {_esc(layer)}</div>\n'
        html += '<div class="grid2">\n'
        for m in sorted(mods, key=lambda x: x.path.name):
            mod_id    = f"{m.layer}/{m.path.name}"
            conf_here = conf_by_module.get(mod_id, [])
            html += _render_module_card(m, conf_here, data.lifecycle)
        html += '</div>\n'

    # YAML inventory
    if data.yaml_files:
        html += '<div class="sl">Fichiers YAML — inventaire clés top-level</div>\n'
        html += '<div class="grid3">\n'
        for yf in data.yaml_files:
            path_str = _esc(str(yf['path']))
            keys_str = ', '.join(_esc(k) for k in yf.get('keys', []))
            html += (f'<div class="card">'
                     f'<div class="card-title">{Path(path_str).name}</div>'
                     f'<div class="note">{path_str}</div>'
                     f'<div class="note" style="margin-top:4px;color:var(--cyan)">{keys_str or "—"}</div>'
                     f'</div>\n')
        html += '</div>\n'

    html += f'''<div style="color:var(--dim);font-size:10px;text-align:center;
padding:24px 0;border-top:1px solid var(--border);margin-top:32px;">
  Généré par generate_architecture.py — {_esc(data.generated_at)}
</div>
</body></html>'''

    return html


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 10 — ENTRYPOINT
# ─────────────────────────────────────────────────────────────────────────────

def collect_all(root_path: str) -> ArchitectureData:
    """
    Orchestre les sections 1→8.
    Retourne ArchitectureData complet.
    """
    root  = Path(root_path).resolve()
    files = scan_files(root_path)

    prc_layers = _detect_prc_layers(files, root)

    modules: List[ModuleInfo] = []
    yaml_files: List[Dict] = []

    for f in files:
        layer = _get_layer(f, root)
        if f.suffix == '.py':
            imports_prc, imports_ext = parse_imports(f, prc_layers)
            doc = parse_docstring(f)
            # Layer depuis @LAYER si renseigné, sinon déduit de l'arborescence
            declared_layer = doc.get('layer', '').strip()
            if declared_layer:
                layer = declared_layer
            modules.append(ModuleInfo(
                path=f,
                layer=layer,
                imports_prc=imports_prc,
                imports_ext=imports_ext,
                doc=doc,
            ))
        elif f.suffix == '.yaml':
            yaml_files.append(parse_yaml_meta(f))
            # Yaml représenté comme ModuleInfo minimal pour scan
            modules.append(ModuleInfo(
                path=f, layer=layer,
                imports_prc=[], imports_ext=[], doc={},
            ))

    py_modules = [m for m in modules if m.path.suffix == '.py']

    graph, graph_violations = build_dependency_graph(py_modules, root)
    topo_order  = topological_sort(py_modules, root)
    lifecycle   = analyze_lifecycle(py_modules)
    conformity  = check_conformity(py_modules, graph_violations)

    return ArchitectureData(
        modules=modules,
        topo_order=topo_order,
        graph=graph,
        lifecycle=lifecycle,
        conformity=conformity,
        yaml_files=yaml_files,
        generated_at=datetime.now().strftime('%Y-%m-%d %H:%M'),
        root=root,
    )


def main() -> None:
    root_path = sys.argv[1] if len(sys.argv) > 1 else '.'
    root      = Path(root_path).resolve()

    print(f"[generate_architecture] Scan : {root}")
    data = collect_all(root_path)

    py_modules   = [m for m in data.modules if m.path.suffix == '.py']
    n_violations = sum(1 for c in data.conformity if c.status == 'VIOLATION')
    n_leaks      = len(data.lifecycle.leaks)
    n_bugs       = sum(len(m.doc.get('bugs', [])) for m in py_modules)
    n_todos      = sum(len(m.doc.get('todos', [])) for m in py_modules)

    print(f"  Modules .py   : {len(py_modules)}")
    print(f"  Violations    : {n_violations}")
    print(f"  Fuites        : {n_leaks}")
    print(f"  Bugs ouverts  : {n_bugs}")
    print(f"  Todos         : {n_todos}")

    # HTML
    docs_dir  = Path(__file__).parent
    html_path = docs_dir / 'architecture.html'
    html      = render_html(data)
    html_path.write_text(html, encoding='utf-8')
    print(f"  → {html_path}")

    # Requirements
    req_path = docs_dir / 'requirements.txt'
    req      = generate_requirements(py_modules)
    req_path.write_text(req, encoding='utf-8')
    print(f"  → {req_path}")

    print("[generate_architecture] OK")


if __name__ == '__main__':
    main()
