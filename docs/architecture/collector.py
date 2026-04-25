# collector.py
from pathlib import Path
from datetime import datetime
from typing import List, Set

from .models import ProjectData, ModuleInfo, YamlInfo, ImportInfo
from . import scanner
from . import ast_parser
from . import docstring_parser
from . import yaml_parser
from .unused_analyzer import analyze_constants, analyze_unused
from .function_detail_builder import build_function_details

def _detect_prc_layers(files: List[Path], root: Path) -> Set[str]:
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

def collect_all(root_path: str, collect_calls: bool = True, collect_vars: bool = True) -> ProjectData:
    root = Path(root_path).resolve()
    files = scanner.scan_files(root_path)

    prc_layers = _detect_prc_layers(files, root)

    modules: List[ModuleInfo] = []
    yaml_files: List[YamlInfo] = []

    for f in files:
        if f.suffix == '.py':
            (imports_prc, imports_ext, constants, functions, calls, metadata,
             all_constants, raw_imports, variables, accesses, objects,
             object_uses, detailed_calls, imports_info, used_names) = ast_parser.parse_file(
                f, prc_layers, collect_calls, collect_vars, collect_flow=True, collect_imports_usage=True
            )
            doc = docstring_parser.parse_docstring(f)
            layer = doc.layer.strip()
            if not layer:
                try:
                    rel = f.relative_to(root)
                    layer = rel.parts[0] if len(rel.parts) > 1 else 'root'
                except ValueError:
                    layer = 'root'
            modules.append(ModuleInfo(
                path=f,
                layer=layer,
                imports_prc=imports_prc,
                imports_ext=imports_ext,
                constants=constants,
                functions=functions,
                calls=calls,
                doc=doc,
                metadata=metadata or {},
                all_constants=all_constants,
                raw_imports=raw_imports,
                variables=variables,
                accesses=accesses,
                objects=objects,
                object_uses=object_uses,
                detailed_calls=detailed_calls,
                imports_info=imports_info,
                used_names=used_names
            ))
        elif f.suffix == '.yaml':
            yaml_info = yaml_parser.parse_yaml(f)
            yaml_files.append(yaml_info)

    modules.sort(key=lambda m: m.path)

    constant_observations = analyze_constants(modules, yaml_files)
    unused_issues = analyze_unused(modules, prc_layers)

    build_function_details(modules, unused_issues)

    return ProjectData(
        root=root,
        modules=modules,
        yaml_files=yaml_files,
        generated_at=datetime.now().strftime('%Y-%m-%d %H:%M'),
        constant_observations=constant_observations,
        unused_issues=unused_issues
    )
