# ast_parser.py
import ast
import sys
from pathlib import Path
from typing import List, Set, Tuple, Optional, Dict, Any
from .models import ConstantInfo, FunctionInfo, CallInfo, VariableInfo, VariableAccess, ObjectInfo, ObjectUse, FunctionCall, ImportInfo

_STDLIB = sys.stdlib_module_names if hasattr(sys, 'stdlib_module_names') else {
    'os', 'sys', 're', 'io', 'abc', 'ast', 'copy', 'math', 'time',
    'json', 'csv', 'gzip', 'zlib', 'enum', 'uuid', 'glob', 'shutil',
    'struct', 'types', 'typing', 'string', 'random', 'hashlib', 'base64',
    'logging', 'warnings', 'argparse', 'pathlib', 'inspect', 'functools',
    'itertools', 'operator', 'contextlib', 'dataclasses', 'collections',
    'threading', 'multiprocessing', 'subprocess', 'concurrent', 'asyncio',
    'socket', 'http', 'urllib', 'email', 'html', 'xml', 'tempfile',
    'traceback', 'datetime', 'calendar', 'locale', 'gettext', 'platform',
    'signal', 'queue', 'heapq', 'bisect', 'array', 'weakref', 'gc',
    'importlib', 'pkgutil', 'textwrap', 'pprint', 'reprlib', 'unittest',
}

def _extract_metadata(node: ast.Assign) -> Optional[Dict[str, Any]]:
    if len(node.targets) != 1:
        return None
    target = node.targets[0]
    if not isinstance(target, ast.Name) or target.id != 'METADATA':
        return None
    if not isinstance(node.value, ast.Dict):
        return None
    try:
        keys = []
        values = []
        for k, v in zip(node.value.keys, node.value.values):
            if not isinstance(k, ast.Constant) or not isinstance(k.value, str):
                return None
            val = ast.literal_eval(v)
            keys.append(k.value)
            values.append(val)
        return dict(zip(keys, values))
    except (ValueError, TypeError, SyntaxError):
        return None

class DataflowCollector(ast.NodeVisitor):
    def __init__(self, module_name: str):
        self.module_name = module_name
        self.current_function = None
        self.objects: List[ObjectInfo] = []
        self.uses: List[ObjectUse] = []
        self.calls: List[FunctionCall] = []
        self._var_counter = 0

    def visit_FunctionDef(self, node):
        old_func = self.current_function
        self.current_function = node.name
        self.generic_visit(node)
        self.current_function = old_func

    def visit_AsyncFunctionDef(self, node):
        old_func = self.current_function
        self.current_function = node.name
        self.generic_visit(node)
        self.current_function = old_func

    def visit_Assign(self, node):
        self.visit(node.value)
        for target in node.targets:
            if isinstance(target, ast.Name):
                self._add_use(target.id, node.lineno, 'write')
        # Ne pas appeler generic_visit

    def visit_AugAssign(self, node):
        if isinstance(node.target, ast.Name):
            self._add_use(node.target.id, node.lineno, 'write')
        self.visit(node.value)

    def visit_Name(self, node):
        if isinstance(node.ctx, ast.Load):
            self._add_use(node.id, node.lineno, 'read')

    def visit_Delete(self, node):
        for target in node.targets:
            if isinstance(target, ast.Name):
                self._add_use(target.id, node.lineno, 'delete')

    def visit_Return(self, node):
        if node.value and isinstance(node.value, ast.Name):
            self._add_use(node.value.id, node.lineno, 'read')
        if node.value:
            self.visit(node.value)

    def visit_Call(self, node):
        func = node.func
        obj_type = None
        if isinstance(func, ast.Name) and func.id in ('list', 'dict', 'set', 'tuple'):
            obj_type = func.id
        elif isinstance(func, ast.Attribute) and isinstance(func.value, ast.Name) and func.value.id == 'np' and func.attr == 'array':
            obj_type = 'np.ndarray'
        if obj_type:
            obj_name = f"anon_{self._var_counter}"
            self._var_counter += 1
            self.objects.append(ObjectInfo(
                name=obj_name,
                creation_line=node.lineno,
                creation_function=self.current_function,
                creation_module=self.module_name,
                obj_type=obj_type,
                is_global=(self.current_function is None)
            ))

        args_names = []
        for arg in node.args:
            if isinstance(arg, ast.Name):
                args_names.append(arg.id)
                self._add_use(arg.id, node.lineno, 'read')
            else:
                args_names.append('literal')
        kw_args = {}
        for kw in node.keywords:
            if kw.value and isinstance(kw.value, ast.Name):
                kw_args[kw.arg] = kw.value.id
                self._add_use(kw.value.id, node.lineno, 'read')
            else:
                kw_args[kw.arg] = 'literal'
        called_module = None
        if isinstance(func, ast.Attribute) and isinstance(func.value, ast.Name):
            called_module = func.value.id
        self.calls.append(FunctionCall(
            caller_module=self.module_name,
            caller_function=self.current_function,
            line=node.lineno,
            called_name=self._get_call_name(node),
            called_module=called_module,
            args=args_names,
            kw_args=kw_args,
            return_var=None
        ))
        self.generic_visit(node)

    def _get_call_name(self, node):
        if isinstance(node.func, ast.Name):
            return node.func.id
        elif isinstance(node.func, ast.Attribute):
            return node.func.attr
        return '?'

    def _add_use(self, name: str, line: int, use_type: str):
        self.uses.append(ObjectUse(
            obj_name=name,
            line=line,
            function=self.current_function,
            use_type=use_type
        ))

def _classify_full(module_fullname: str, prc_layers: Set[str],
                   prc_imports: List[str], ext_imports: List[str]) -> None:
    if not module_fullname:
        return
    pkg = module_fullname.split('.')[0]
    if pkg in prc_layers:
        prc_imports.append(module_fullname)
    elif pkg not in _STDLIB and pkg not in ('', '__future__'):
        ext_imports.append(pkg)

def _extract_constant(node: ast.Assign) -> Optional[ConstantInfo]:
    if len(node.targets) != 1:
        return None
    target = node.targets[0]
    if not isinstance(target, ast.Name):
        return None
    name = target.id
    value_node = node.value
    if isinstance(value_node, ast.Constant):
        if isinstance(value_node.value, (int, float, str)):
            return ConstantInfo(
                name=name,
                value=repr(value_node.value),
                line=node.lineno
            )
    return None

def _extract_function(node: ast.FunctionDef) -> FunctionInfo:
    args = node.args
    pos_args = [a.arg for a in args.args]
    kw_args = [a.arg for a in args.kwonlyargs]
    var_arg = args.vararg.arg if args.vararg else None
    kw_arg = args.kwarg.arg if args.kwarg else None
    parameters = []
    parameters.extend(pos_args)
    if var_arg:
        parameters.append(f'*{var_arg}')
    parameters.extend(kw_args)
    if kw_arg:
        parameters.append(f'**{kw_arg}')
    parts = []
    if pos_args:
        parts.append(', '.join(pos_args))
    if var_arg:
        parts.append(f'*{var_arg}')
    if kw_args:
        parts.append(', '.join(kw_args))
    if kw_arg:
        parts.append(f'**{kw_arg}')
    signature = f"def {node.name}({', '.join(parts)})"
    decorators = [d.id if isinstance(d, ast.Name) else repr(d) for d in node.decorator_list]
    return FunctionInfo(
        name=node.name,
        line=node.lineno,
        signature=signature,
        decorators=decorators,
        parameters=parameters
    )

class CallCollector(ast.NodeVisitor):
    def __init__(self, module_name: str):
        self.module_name = module_name
        self.current_function = None
        self.calls: List[CallInfo] = []

    def visit_FunctionDef(self, node: ast.FunctionDef):
        old_function = self.current_function
        self.current_function = node.name
        self.generic_visit(node)
        self.current_function = old_function

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef):
        old_function = self.current_function
        self.current_function = node.name
        self.generic_visit(node)
        self.current_function = old_function

    def visit_Call(self, node: ast.Call):
        func = node.func
        if isinstance(func, ast.Name):
            func_name = func.id
            resolved = None
        elif isinstance(func, ast.Attribute):
            if isinstance(func.value, ast.Name):
                resolved = func.value.id
                func_name = func.attr
            else:
                func_name = func.attr
                resolved = None
        else:
            self.generic_visit(node)
            return
        call = CallInfo(
            caller_module=self.module_name,
            caller_function=self.current_function,
            line=node.lineno,
            function_name=func_name,
            resolved_module=resolved
        )
        self.calls.append(call)
        self.generic_visit(node)

def _collect_all_calls(tree: ast.AST, module_name: str) -> List[CallInfo]:
    collector = CallCollector(module_name)
    collector.visit(tree)
    return collector.calls

class VariableCollector(ast.NodeVisitor):
    def __init__(self, module_name: str):
        self.module_name = module_name
        self.current_function = None
        self.variables: List[VariableInfo] = []
        self.accesses: List[VariableAccess] = []
        self._seen_defs: Set[Tuple[Optional[str], str]] = set()
        self._returned_vars: Set[str] = set()

    def visit_FunctionDef(self, node: ast.FunctionDef):
        old_function = self.current_function
        self.current_function = node.name
        self._returned_vars = set()
        args = node.args
        for arg in args.args:
            self._add_variable(arg.arg, arg.lineno, is_param=True)
        if args.vararg:
            self._add_variable(args.vararg.arg, args.vararg.lineno, is_param=True)
        if args.kwarg:
            self._add_variable(args.kwarg.arg, args.kwarg.lineno, is_param=True)
        self.generic_visit(node)
        for var in self.variables:
            if var.function == node.name and var.name in self._returned_vars:
                var.is_returned = True
        self.current_function = old_function

    def visit_Return(self, node: ast.Return):
        if node.value and isinstance(node.value, ast.Name):
            self._returned_vars.add(node.value.id)
        self.generic_visit(node)

    def visit_Assign(self, node: ast.Assign):
        for target in node.targets:
            if isinstance(target, ast.Name):
                self._add_access(target.id, target.lineno, 'write')
                self._add_variable(target.id, target.lineno)
        self.generic_visit(node.value)

    def visit_AugAssign(self, node: ast.AugAssign):
        if isinstance(node.target, ast.Name):
            self._add_access(node.target.id, node.lineno, 'write')
            self._add_variable(node.target.id, node.lineno)
        self.generic_visit(node.value)

    def visit_Name(self, node: ast.Name):
        if isinstance(node.ctx, ast.Load):
            self._add_access(node.id, node.lineno, 'read')
        self.generic_visit(node)

    def visit_Delete(self, node: ast.Delete):
        for target in node.targets:
            if isinstance(target, ast.Name):
                self._add_access(target.id, target.lineno, 'delete')
        self.generic_visit(node)

    def visit_Call(self, node: ast.Call):
        for arg in node.args:
            if isinstance(arg, ast.Name):
                self._add_access(arg.id, arg.lineno, 'read')
        for kw in node.keywords:
            if kw.value and isinstance(kw.value, ast.Name):
                self._add_access(kw.value.id, kw.value.lineno, 'read')
        self.generic_visit(node)

    def _add_variable(self, name: str, line: int, is_param: bool = False):
        key = (self.current_function, name)
        if key not in self._seen_defs:
            self._seen_defs.add(key)
            scope = 'global' if self.current_function is None else 'local'
            self.variables.append(VariableInfo(
                name=name,
                line_def=line,
                scope=scope,
                function=self.current_function,
                is_param=is_param
            ))
        if is_param:
            self._add_access(name, line, 'write')

    def _add_access(self, name: str, line: int, access_type: str):
        self.accesses.append(VariableAccess(
            name=name,
            line=line,
            access_type=access_type,
            function=self.current_function
        ))

def _extract_all_constants(tree: ast.AST) -> List[ConstantInfo]:
    constants = []
    class ConstantVisitor(ast.NodeVisitor):
        def visit_Constant(self, node: ast.Constant):
            if isinstance(node.value, (int, float, str, bool)):
                constants.append(ConstantInfo(
                    name='',
                    value=repr(node.value),
                    line=node.lineno
                ))
            self.generic_visit(node)
    ConstantVisitor().visit(tree)
    return constants

class NameUsageCollector(ast.NodeVisitor):
    """Collecte tous les noms utilisés en lecture (Load)."""
    def __init__(self):
        self.used_names = set()

    def visit_Name(self, node):
        if isinstance(node.ctx, ast.Load):
            self.used_names.add(node.id)
        self.generic_visit(node)

    def visit_Attribute(self, node):
        # Pour les attributs, on collecte le nom de l'objet (ex: np.array -> 'np')
        if isinstance(node.value, ast.Name):
            self.used_names.add(node.value.id)
        self.generic_visit(node)

def _collect_imports(tree) -> List[ImportInfo]:
    imports = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                imports.append(ImportInfo(
                    module=alias.name,
                    name=alias.name,
                    alias=alias.asname,
                    line=node.lineno,
                    is_from=False
                ))
        elif isinstance(node, ast.ImportFrom) and node.module:
            for alias in node.names:
                imports.append(ImportInfo(
                    module=node.module,
                    name=alias.name,
                    alias=alias.asname,
                    line=node.lineno,
                    is_from=True
                ))
    return imports

def parse_file(
    filepath: Path,
    prc_layers: Set[str],
    collect_calls: bool = False,
    collect_vars: bool = False,
    collect_flow: bool = False,
    collect_imports_usage: bool = False
) -> Tuple[List[str], List[str], List[ConstantInfo], List[FunctionInfo], List[CallInfo], Optional[Dict[str, Any]], List[ConstantInfo], List[str], List[VariableInfo], List[VariableAccess], List[ObjectInfo], List[ObjectUse], List[FunctionCall], List[ImportInfo], Set[str]]:
    try:
        source = filepath.read_text(encoding='utf-8', errors='replace')
        tree = ast.parse(source, filename=str(filepath))
    except SyntaxError:
        return [], [], [], [], [], None, [], [], [], [], [], [], [], [], set()

    imports_prc: List[str] = []
    imports_ext: List[str] = []
    constants: List[ConstantInfo] = []
    functions: List[FunctionInfo] = []
    metadata = None
    raw_imports: List[str] = []
    calls: List[CallInfo] = []
    variables: List[VariableInfo] = []
    accesses: List[VariableAccess] = []
    objects: List[ObjectInfo] = []
    object_uses: List[ObjectUse] = []
    detailed_calls: List[FunctionCall] = []

    module_name = filepath.stem

    for node in ast.iter_child_nodes(tree):
        if isinstance(node, ast.Import) or isinstance(node, ast.ImportFrom):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    fullname = alias.name
                    raw_imports.append(fullname)
                    _classify_full(fullname, prc_layers, imports_prc, imports_ext)
            else:
                if node.module:
                    fullname = node.module
                    raw_imports.append(fullname)
                    _classify_full(fullname, prc_layers, imports_prc, imports_ext)
        elif isinstance(node, ast.Assign):
            const = _extract_constant(node)
            if const:
                constants.append(const)
            meta = _extract_metadata(node)
            if meta is not None:
                metadata = meta
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            functions.append(_extract_function(node))

    all_constants = _extract_all_constants(tree)

    if collect_calls:
        calls = _collect_all_calls(tree, module_name)

    if collect_vars:
        collector = VariableCollector(module_name)
        collector.visit(tree)
        variables = collector.variables
        accesses = collector.accesses

    if collect_flow:
        flow_collector = DataflowCollector(module_name)
        flow_collector.visit(tree)
        objects = flow_collector.objects
        object_uses = flow_collector.uses
        detailed_calls = flow_collector.calls

    imports_info = []
    used_names = set()
    if collect_imports_usage:
        imports_info = _collect_imports(tree)
        name_collector = NameUsageCollector()
        name_collector.visit(tree)
        used_names = name_collector.used_names

    return (
        sorted(set(imports_prc)),
        sorted(set(imports_ext)),
        constants,
        functions,
        calls,
        metadata,
        all_constants,
        sorted(set(raw_imports)),
        variables,
        accesses,
        objects,
        object_uses,
        detailed_calls,
        imports_info,
        used_names
    )
