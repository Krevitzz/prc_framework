# models.py
from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Dict, Optional, Any, Set

@dataclass
class ConstantInfo:
    name: str
    value: str
    line: int

@dataclass
class FunctionInfo:
    name: str
    line: int
    signature: str
    decorators: List[str] = field(default_factory=list)
    parameters: List[str] = field(default_factory=list)

@dataclass
class VariableInfo:
    name: str
    line_def: int
    scope: str
    function: Optional[str] = None
    is_param: bool = False
    is_returned: bool = False

@dataclass
class VariableAccess:
    name: str
    line: int
    access_type: str
    function: Optional[str] = None

@dataclass
class VariableDetail:
    name: str
    is_global: bool = False
    is_param: bool = False
    is_returned: bool = False
    is_unused: bool = False
    is_constant: bool = False

@dataclass
class FunctionDetail:
    name: str
    calls_internal: List[str] = field(default_factory=list)
    calls_external: List[str] = field(default_factory=list)
    reads: List[VariableDetail] = field(default_factory=list)
    writes: List[VariableDetail] = field(default_factory=list)
    deletes: List[VariableDetail] = field(default_factory=list)

@dataclass
class ObjectInfo:
    name: str
    creation_line: int
    creation_function: Optional[str]
    creation_module: str
    obj_type: Optional[str]
    is_global: bool = False

@dataclass
class ObjectUse:
    obj_name: str
    line: int
    function: Optional[str]
    use_type: str

@dataclass
class FunctionCall:
    caller_module: str
    caller_function: Optional[str]
    line: int
    called_name: str
    called_module: Optional[str]
    args: List[str]
    kw_args: Dict[str, str]
    return_var: Optional[str]

@dataclass
class CallInfo:
    caller_module: str
    line: int
    function_name: str
    caller_function: Optional[str] = None
    resolved_module: Optional[str] = None

@dataclass
class ImportInfo:
    module: str
    name: str
    alias: Optional[str]
    line: int
    is_from: bool

@dataclass
class DocstringSections:
    role: str = ""
    layer: str = ""

@dataclass
class ModuleInfo:
    path: Path
    layer: str
    imports_prc: List[str]
    imports_ext: List[str]
    constants: List[ConstantInfo]
    functions: List[FunctionInfo]
    calls: List[CallInfo]
    doc: DocstringSections
    metadata: Dict[str, Any] = field(default_factory=dict)
    all_constants: List[ConstantInfo] = field(default_factory=list)
    raw_imports: List[str] = field(default_factory=list)
    variables: List[VariableInfo] = field(default_factory=list)
    accesses: List[VariableAccess] = field(default_factory=list)
    objects: List[ObjectInfo] = field(default_factory=list)
    object_uses: List[ObjectUse] = field(default_factory=list)
    detailed_calls: List[FunctionCall] = field(default_factory=list)
    function_details: Dict[str, FunctionDetail] = field(default_factory=dict)
    imports_info: List[ImportInfo] = field(default_factory=list)
    used_names: Set[str] = field(default_factory=set)

@dataclass
class UnusedIssue:
    kind: str
    module: str
    name: str
    line: Optional[int] = None
    function: Optional[str] = None
    extra: Optional[Dict[str, Any]] = None

@dataclass
class YamlInfo:
    path: Path
    top_keys: List[str]
    scalar_values: Set[Any] = field(default_factory=set)
    description: str = "" 

@dataclass
class ProjectData:
    root: Path
    modules: List[ModuleInfo]
    yaml_files: List[YamlInfo]
    generated_at: str
    constant_observations: List[Dict[str, Any]] = field(default_factory=list)
    unused_issues: List[UnusedIssue] = field(default_factory=list)
