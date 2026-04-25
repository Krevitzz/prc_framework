# unused_analyzer.py
from typing import List, Set, Dict, Optional, Any
from collections import defaultdict
from .models import ModuleInfo, YamlInfo, UnusedIssue, ImportInfo
from . import scanner

def analyze_constants(modules: List[ModuleInfo], yaml_files: List[YamlInfo]) -> List[Dict[str, Any]]:
    """Retourne les constantes numériques non présentes dans les YAML."""
    yaml_values = set()
    for yf in yaml_files:
        yaml_values.update(yf.scalar_values)

    observations = []
    for module in modules:
        if module.path.suffix != '.py':
            continue
        for const in module.all_constants:
            try:
                val = eval(const.value)
                if not isinstance(val, (int, float)):
                    continue
            except:
                continue
            if val not in yaml_values:
                observations.append({
                    'module': f"{module.layer}/{module.path.stem}",
                    'line': const.line,
                    'name': const.name or '(literal)',
                    'value': const.value,
                    'type': 'unmatched_constant'
                })
    return observations

def analyze_unused_imports(module: ModuleInfo) -> List[UnusedIssue]:
    """Détecte les imports dont l'alias local n'est jamais utilisé."""
    issues = []
    mod_id = f"{module.layer}/{module.path.stem}"
    used = module.used_names

    for imp in module.imports_info:
        # Déterminer le nom local utilisé dans le code
        if imp.is_from:
            local_name = imp.alias if imp.alias else imp.name
        else:
            local_name = imp.alias if imp.alias else imp.module.split('.')[-1]
        # Ignorer les imports de type '*' (from module import *)
        if local_name == '*':
            continue
        if local_name not in used:
            issues.append(UnusedIssue(
                kind='import',
                module=mod_id,
                name=f"{imp.module}:{imp.name}" + (f" as {imp.alias}" if imp.alias else ""),
                line=imp.line,
                function=None,
                extra={'local_name': local_name}
            ))
    return issues

def analyze_unused(modules: List[ModuleInfo], prc_layers: Set[str]) -> List[UnusedIssue]:
    issues = []

    # ----- 1. Modules non importés -----
    all_modules = set()
    imported_modules = set()
    for m in modules:
        if m.path.suffix != '.py':
            continue
        mod_id = f"{m.layer}/{m.path.stem}"
        all_modules.add(mod_id)
        for imp in m.imports_prc:
            imported_modules.add(imp.replace('.', '/'))

    dead_modules = all_modules - imported_modules
    for mod_id in dead_modules:
        parts = mod_id.split('/')
        if len(parts) != 2:
            continue
        layer, name = parts
        if scanner.should_exclude_module(mod_id, layer, name):
            continue
        issues.append(UnusedIssue(
            kind='module',
            module=mod_id,
            name=mod_id,
            line=None,
            function=None,
            extra=None
        ))

    # ----- 2. Fonctions non appelées -----
    defined_functions = []          # (mod_id, func_name, line)
    called_funcs = set()

    for m in modules:
        if m.path.suffix != '.py':
            continue
        mod_id = f"{m.layer}/{m.path.stem}"
        for f in m.functions:
            defined_functions.append((mod_id, f.name, f.line))
        for call in m.calls:
            if call.resolved_module is None or call.resolved_module in prc_layers:
                called_funcs.add(call.function_name)
        for dcall in m.detailed_calls:
            if dcall.called_module is None or dcall.called_module in prc_layers:
                called_funcs.add(dcall.called_name)

    for mod_id, func_name, line in defined_functions:
        if scanner.should_exclude_module(mod_id, *mod_id.split('/')):
            continue
        if scanner.should_exclude_function(func_name):
            continue
        if func_name not in called_funcs:
            issues.append(UnusedIssue(
                kind='function',
                module=mod_id,
                name=func_name,
                line=line,
                function=None,
                extra=None
            ))

    # ----- 3. Variables et objets inutilisés -----
    for m in modules:
        if m.path.suffix != '.py':
            continue
        mod_id = f"{m.layer}/{m.path.stem}"
        if scanner.should_exclude_module(mod_id, m.layer, m.path.stem):
            continue

        param_names = {v.name for v in m.variables if v.is_param}

        # Variables : écrites mais jamais lues
        reads = set()
        writes = set()
        write_lines = {}
        write_funcs = {}
        for acc in m.accesses:
            if acc.access_type == 'read':
                reads.add(acc.name)
            elif acc.access_type == 'write':
                writes.add(acc.name)
                if acc.name not in write_lines:
                    write_lines[acc.name] = acc.line
                    write_funcs[acc.name] = acc.function
        for var in writes:
            if var not in reads:
                if var in param_names:
                    continue
                if len(var) == 1:
                    continue
                if scanner.is_constant_name(var):
                    continue
                issues.append(UnusedIssue(
                    kind='variable',
                    module=mod_id,
                    name=var,
                    line=write_lines.get(var),
                    function=write_funcs.get(var),
                    extra=None
                ))

        # Objets : créés mais jamais utilisés
        used_objs = set()
        for use in m.object_uses:
            used_objs.add(use.obj_name)
        for obj in m.objects:
            if obj.name not in used_objs:
                issues.append(UnusedIssue(
                    kind='object',
                    module=mod_id,
                    name=obj.name,
                    line=obj.creation_line,
                    function=obj.creation_function,
                    extra={'obj_type': obj.obj_type, 'is_global': obj.is_global}
                ))

        # Imports inutilisés
        issues.extend(analyze_unused_imports(m))

    return issues
