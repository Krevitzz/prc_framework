# docstring_parser.py
import ast
from pathlib import Path
from typing import List, Dict, Optional

from .models import DocstringSections

KNOWN_SECTIONS = {
    '@ROLE', '@LAYER', '@EXPORTS', '@LIFECYCLE',
    '@CONFORMITY', '@BUGS', '@TODOS', '@QUESTIONS'
}


def _parse_section_lines(lines: List[str], start: int) -> List[str]:
    """Collecte les lignes après un marqueur jusqu'au prochain marqueur."""
    result = []
    i = start + 1
    while i < len(lines):
        line = lines[i]
        stripped = line.strip()
        if stripped and stripped.split()[0].upper() in KNOWN_SECTIONS:
            break
        if stripped:
            result.append(stripped)
        i += 1
    return result


def _parse_exports(lines: List[str]) -> List[Dict]:
    exports = []
    for line in lines:
        if '→' in line or '->' in line:
            sep = '→' if '→' in line else '->'
            parts = line.split(sep, 1)
            sig = parts[0].strip()
            rest = parts[1].strip() if len(parts) > 1 else ''
            note_parts = rest.split('|', 1)
            ret = note_parts[0].strip()
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
                'verb': parts[0].upper(),
                'object': parts[1],
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
            'rule': rule_note[0].strip(),
            'note': rule_note[1].strip() if len(rule_note) > 1 else '',
        })
    return items


def _parse_bugs(lines: List[str]) -> List[Dict]:
    bugs = []
    for line in lines:
        if '—' in line or '-' in line:
            sep = '—' if '—' in line else ' - '
            parts = line.split(sep, 1)
            bugs.append({
                'id': parts[0].strip(),
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
            b_end = l.index(']', b_start)
            blocks = l[b_start+8:b_end].strip()
            l = l[:b_start] + l[b_end+1:]
        if '—' in l or ' - ' in l:
            sep = '—' if '—' in l else ' - '
            parts = l.split(sep, 1)
            todos.append({
                'id': parts[0].strip(),
                'blocks': blocks,
                'desc': parts[1].strip() if len(parts) > 1 else '',
            })
        elif l.strip():
            todos.append({'id': '', 'blocks': blocks, 'desc': l.strip()})
    return todos


def _parse_questions(lines: List[str]) -> List[Dict]:
    qs = []
    for line in lines:
        if '—' in line or ' - ' in line:
            sep = '—' if '—' in line else ' - '
            parts = line.split(sep, 1)
            qs.append({
                'id': parts[0].strip(),
                'desc': parts[1].strip() if len(parts) > 1 else '',
            })
    return qs


def parse_docstring(filepath: Path) -> DocstringSections:
    """
    Extrait et parse la docstring d'un fichier .py.
    Retourne un objet DocstringSections (toutes les sections, vides si absentes).
    """
    empty = DocstringSections()
    try:
        source = filepath.read_text(encoding='utf-8', errors='replace')
        tree = ast.parse(source)
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
        if not stripped:
            continue
        token = stripped.split()[0].upper() if stripped.split() else ''
        if token in KNOWN_SECTIONS:
            current_section = token
            # Valeur inline sur la même ligne
            rest = stripped[len(token):].strip()
            sections[current_section] = [rest] if rest else []
            # Ajouter les lignes suivantes
            sections[current_section] += _parse_section_lines(lines, i)
        # Les lignes hors section sont ignorées (texte libre)

    result = DocstringSections()
    result.role = ' '.join(sections.get('@ROLE', [])).strip()
    result.layer = ' '.join(sections.get('@LAYER', [])).strip()
    result.exports = _parse_exports(sections.get('@EXPORTS', []))
    result.lifecycle = _parse_lifecycle(sections.get('@LIFECYCLE', []))
    result.conformity = _parse_conformity(sections.get('@CONFORMITY', []))
    result.bugs = _parse_bugs(sections.get('@BUGS', []))
    result.todos = _parse_todos(sections.get('@TODOS', []))
    result.questions = _parse_questions(sections.get('@QUESTIONS', []))
    return result
