# function_detail_builder.py
from typing import Dict, Set, List
from .models import ModuleInfo, UnusedIssue, FunctionDetail, VariableDetail

def build_function_details(modules: List[ModuleInfo], unused_issues: List[UnusedIssue]) -> None:
    # Index des variables inutilisées par module
    unused_by_module: Dict[str, Set[str]] = {}
    for issue in unused_issues:
        if issue.kind == 'variable':
            unused_by_module.setdefault(issue.module, set()).add(issue.name)

    for module in modules:
        if module.path.suffix != '.py':
            continue
        module_id = f"{module.layer}/{module.path.stem}"
        unused_vars = unused_by_module.get(module_id, set())
        global_vars = {v.name for v in module.variables if v.scope == 'global'}

        # Index des informations de variables par (fonction, nom)
        var_info = {}
        for v in module.variables:
            key = (v.function, v.name)
            var_info[key] = v

        # Pour chaque fonction
        for func in module.functions:
            fd = FunctionDetail(name=func.name)

            # Appels internes/externes
            internal = set()
            external = set()
            for call in module.calls:
                if call.caller_function == func.name:
                    if call.resolved_module and call.resolved_module in module.imports_prc:
                        internal.add(call.function_name)
                    else:
                        if call.resolved_module:
                            external.add(f"{call.resolved_module}.{call.function_name}")
                        else:
                            external.add(call.function_name)
            fd.calls_internal = sorted(internal)
            fd.calls_external = sorted(external)

            # Accès aux variables
            reads_dict = {}
            writes_dict = {}
            deletes_dict = {}
            for acc in module.accesses:
                if acc.function != func.name:
                    continue
                # Vérifier si la variable est pertinente (globale, paramètre, retournée)
                key = (func.name, acc.name)
                info = var_info.get(key)
                is_global = acc.name in global_vars
                is_param = info.is_param if info else False
                is_returned = info.is_returned if info else False
                is_unused = acc.name in unused_vars
                is_constant = acc.name.isupper()   # 👈 règle simple

                # On ne garde que les variables pertinentes
                if not (is_global or is_param or is_returned):
                    continue

                detail = VariableDetail(
                    name=acc.name,
                    is_global=is_global,
                    is_param=is_param,
                    is_returned=is_returned,
                    is_unused=is_unused,
                    is_constant=is_constant
                )
                if acc.access_type == 'read':
                    reads_dict[acc.name] = detail
                elif acc.access_type == 'write':
                    writes_dict[acc.name] = detail
                elif acc.access_type == 'delete':
                    deletes_dict[acc.name] = detail

            fd.reads = sorted(reads_dict.values(), key=lambda d: d.name)
            fd.writes = sorted(writes_dict.values(), key=lambda d: d.name)
            fd.deletes = sorted(deletes_dict.values(), key=lambda d: d.name)

            module.function_details[func.name] = fd
