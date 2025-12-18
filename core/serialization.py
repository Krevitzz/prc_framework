"""
core/serialization.py

Import/export pour le framework PRC.

RESPONSABILITÉS:
- Sauvegarder/charger InformationSpace (format JSON)
- Sauvegarder/charger états de simulation
- Sauvegarder/charger historiques
- Gérer la compatibilité des formats

FORMATS STANDARDISÉS:
- information_space.json : Encodage D^(X)
- simulation_state.json : État à un instant donné
- simulation_history.json : Trajectoire complète
- experiment_spec.json : Spécification d'expérience (pour plugins)
"""

import json
import numpy as np
from pathlib import Path
from typing import Dict, Any, Optional, Union, List
from datetime import datetime

from .information_space import InformationSpace


# ============================================================================
# CONVERSION NUMPY ↔ JSON
# ============================================================================

class NumpyEncoder(json.JSONEncoder):
    """
    Encodeur JSON custom pour gérer les types NumPy.
    
    Convertit:
    - np.ndarray → list
    - np.integer → int
    - np.floating → float
    - np.bool_ → bool
    """
    
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.bool_):
            return bool(obj)
        return super().default(obj)


def numpy_to_json_compatible(obj: Any) -> Any:
    """
    Convertit récursivement les types NumPy en types Python natifs.
    
    Args:
        obj: Objet à convertir (peut être dict, list, etc.)
    
    Returns:
        Objet avec types JSON-compatibles
    """
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.bool_):
        return bool(obj)
    elif isinstance(obj, dict):
        return {k: numpy_to_json_compatible(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [numpy_to_json_compatible(item) for item in obj]
    else:
        return obj


# ============================================================================
# INFORMATION SPACE (encodages)
# ============================================================================

def save_information_space(space: InformationSpace, 
                          filepath: Union[str, Path],
                          include_metadata: bool = True) -> None:
    """
    Sauvegarde un InformationSpace en JSON.
    
    Format:
    {
        "version": "1.0",
        "timestamp": "2024-12-17T...",
        "n_dof": 100,
        "C": [[...], [...], ...],
        "metadata": {...}
    }
    
    Args:
        space: InformationSpace à sauvegarder
        filepath: Chemin du fichier de sortie
        include_metadata: Si False, omet les métadonnées (fichier plus léger)
    """
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    
    data = {
        "version": "1.0",
        "format": "prc_information_space",
        "timestamp": datetime.now().isoformat(),
        "n_dof": space.n_dof,
        "C": space.C.tolist()
    }
    
    if include_metadata:
        data["metadata"] = numpy_to_json_compatible(space.metadata)
    
    with open(filepath, 'w') as f:
        json.dump(data, f, indent=2)


def load_information_space(filepath: Union[str, Path]) -> InformationSpace:
    """
    Charge un InformationSpace depuis JSON.
    
    Args:
        filepath: Chemin du fichier
    
    Returns:
        InformationSpace reconstruit
    
    Raises:
        ValueError: Si format incompatible
    """
    with open(filepath, 'r') as f:
        data = json.load(f)
    
    # Vérification du format
    if data.get("format") != "prc_information_space":
        raise ValueError(f"Format incompatible: {data.get('format')}")
    
    # Reconstruction
    C = np.array(data["C"])
    metadata = data.get("metadata", {})
    
    return InformationSpace(C, metadata)


# ============================================================================
# SIMULATION STATE (snapshots)
# ============================================================================

def save_simulation_state(kernel, filepath: Union[str, Path]) -> None:
    """
    Sauvegarde l'état actuel d'une simulation.
    
    Format:
    {
        "iteration": 1000,
        "C_current": [[...], ...],
        "gamma_parameters": {...},
        "initial_state": {...},
        "trajectory_summary": {...}
    }
    
    Args:
        kernel: Instance de PRCKernel
        filepath: Chemin du fichier de sortie
    """
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    
    data = {
        "version": "1.0",
        "format": "prc_simulation_state",
        "timestamp": datetime.now().isoformat(),
        "iteration": kernel.iteration,
        "n_dof": kernel.D_0.n_dof,
        "C_current": kernel.C_current.tolist(),
        "gamma": kernel.gamma.get_parameters(),
        "initial_metadata": numpy_to_json_compatible(kernel.D_0.metadata),
        "trajectory_summary": numpy_to_json_compatible(
            kernel.get_trajectory_summary()
        )
    }
    
    with open(filepath, 'w') as f:
        json.dump(data, f, indent=2)


def load_simulation_state(filepath: Union[str, Path]) -> Dict[str, Any]:
    """
    Charge un état de simulation.
    
    Note: Ne reconstruit pas le kernel (car Γ peut ne pas être sérialisable).
    Retourne les données pour reconstruction manuelle.
    
    Args:
        filepath: Chemin du fichier
    
    Returns:
        Dictionnaire avec les données de l'état
    """
    with open(filepath, 'r') as f:
        data = json.load(f)
    
    if data.get("format") != "prc_simulation_state":
        raise ValueError(f"Format incompatible: {data.get('format')}")
    
    # Convertit C en numpy
    data["C_current"] = np.array(data["C_current"])
    
    return data


# ============================================================================
# SIMULATION HISTORY (trajectoires)
# ============================================================================

def save_simulation_history(kernel, 
                           filepath: Union[str, Path],
                           compress: bool = False) -> None:
    """
    Sauvegarde l'historique complet d'une simulation.
    
    ATTENTION: Peut produire des fichiers très volumineux.
    
    Format:
    {
        "iterations": [0, 1, 2, ...],
        "history": [
            {"iteration": 0, "C": [[...], ...]},
            {"iteration": 1, "C": [[...], ...]},
            ...
        ],
        "metadata": {...}
    }
    
    Args:
        kernel: Instance de PRCKernel avec historique
        filepath: Chemin du fichier de sortie
        compress: Si True, sauvegarde en format compressé (non implémenté)
    """
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    
    history = kernel.get_history()
    
    data = {
        "version": "1.0",
        "format": "prc_simulation_history",
        "timestamp": datetime.now().isoformat(),
        "n_dof": kernel.D_0.n_dof,
        "n_snapshots": len(history),
        "gamma": kernel.gamma.get_parameters(),
        "initial_metadata": numpy_to_json_compatible(kernel.D_0.metadata),
        "history": [
            {
                "iteration": state.iteration,
                "C": state.C.tolist()
            }
            for state in history
        ]
    }
    
    # Avertissement si fichier volumineux
    estimated_size_mb = (data["n_dof"]**2 * data["n_snapshots"] * 8) / (1024**2)
    if estimated_size_mb > 100:
        print(f"ATTENTION: Taille estimée du fichier: {estimated_size_mb:.1f} MB")
    
    with open(filepath, 'w') as f:
        json.dump(data, f, indent=2 if not compress else None)


def load_simulation_history(filepath: Union[str, Path]) -> Dict[str, Any]:
    """
    Charge un historique de simulation.
    
    Args:
        filepath: Chemin du fichier
    
    Returns:
        Dictionnaire avec métadonnées + liste de matrices C
    """
    with open(filepath, 'r') as f:
        data = json.load(f)
    
    if data.get("format") != "prc_simulation_history":
        raise ValueError(f"Format incompatible: {data.get('format')}")
    
    # Convertit les matrices C en numpy
    for state in data["history"]:
        state["C"] = np.array(state["C"])
    
    return data


# ============================================================================
# EXPERIMENT SPECIFICATIONS (pour plugins)
# ============================================================================

def load_experiment_spec(filepath: Union[str, Path]) -> Dict[str, Any]:
    """
    Charge une spécification d'expérience (format standard pour plugins).
    
    Format attendu:
    {
        "domain": "quantum" | "gravity" | ...,
        "experiment": "double_slit" | "schwarzschild" | ...,
        "system": {
            "n_dof": 100,
            "dof_description": "...",
            ...
        },
        "initial_configuration": {...},
        "encoding_parameters": {...},
        "expected_signatures": {...}
    }
    
    Args:
        filepath: Chemin du fichier JSON
    
    Returns:
        Dictionnaire avec spécifications
    
    Raises:
        ValueError: Si champs obligatoires manquants
    """
    with open(filepath, 'r') as f:
        spec = json.load(f)
    
    # Validation des champs obligatoires
   required_fields = ["domain", "experiment", "system"]
    missing = [f for f in required_fields if f not in spec]
    if missing:
        raise ValueError(f"Champs manquants dans spec: {missing}")
    
    return spec


def save_experiment_spec(spec: Dict[str, Any], 
                        filepath: Union[str, Path]) -> None:
    """
    Sauvegarde une spécification d'expérience.
    
    Args:
        spec: Dictionnaire de spécifications
        filepath: Chemin du fichier de sortie
    """
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    
    # Ajoute métadonnées de version
    spec_with_meta = {
        "version": "1.0",
        "format": "prc_experiment_spec",
        "timestamp": datetime.now().isoformat(),
        **spec
    }
    
    with open(filepath, 'w') as f:
        json.dump(spec_with_meta, f, indent=2)


# ============================================================================
# BATCH OPERATIONS (utilitaires pour exports multiples)
# ============================================================================

def save_batch_results(results: List[Dict[str, Any]],
                      output_dir: Union[str, Path],
                      prefix: str = "result") -> None:
    """
    Sauvegarde plusieurs résultats en batch.
    
    Crée un fichier par résultat: {prefix}_000.json, {prefix}_001.json, ...
    
    Args:
        results: Liste de dictionnaires à sauvegarder
        output_dir: Dossier de sortie
        prefix: Préfixe pour les noms de fichiers
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    for i, result in enumerate(results):
        filename = f"{prefix}_{i:03d}.json"
        filepath = output_dir / filename
        
        with open(filepath, 'w') as f:
            json.dump(numpy_to_json_compatible(result), f, indent=2)


def load_batch_results(input_dir: Union[str, Path],
                      pattern: str = "result_*.json") -> List[Dict[str, Any]]:
    """
    Charge tous les résultats d'un dossier.
    
    Args:
        input_dir: Dossier contenant les fichiers
        pattern: Pattern de fichiers (glob)
    
    Returns:
        Liste de dictionnaires chargés
    """
    input_dir = Path(input_dir)
    files = sorted(input_dir.glob(pattern))
    
    results = []
    for filepath in files:
        with open(filepath, 'r') as f:
            results.append(json.load(f))
    
    return results


# ============================================================================
# EXPORT WRAPPERS (pour kernel.py)
# ============================================================================

def export_state(kernel, filepath: Union[str, Path]) -> None:
    """
    Export l'état actuel du kernel.
    
    Wrapper pour save_simulation_state() avec signature simplifiée.
    """
    save_simulation_state(kernel, filepath)


def export_history(kernel, filepath: Union[str, Path]) -> None:
    """
    Export l'historique du kernel.
    
    Wrapper pour save_simulation_history() avec signature simplifiée.
    """
    save_simulation_history(kernel, filepath)


# ============================================================================
# VALIDATION ET MIGRATION DE FORMAT
# ============================================================================

def validate_format(filepath: Union[str, Path]) -> Dict[str, Any]:
    """
    Valide un fichier PRC et retourne ses métadonnées.
    
    Args:
        filepath: Chemin du fichier à valider
    
    Returns:
        {
            "valid": bool,
            "format": str,
            "version": str,
            "errors": List[str]  # si invalid
        }
    """
    try:
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        format_type = data.get("format", "unknown")
        version = data.get("version", "unknown")
        
        # Vérifications basiques selon le format
        errors = []
        
        if format_type == "prc_information_space":
            if "C" not in data:
                errors.append("Champ 'C' manquant")
            elif not isinstance(data["C"], list):
                errors.append("'C' doit être une liste")
        
        elif format_type == "prc_simulation_state":
            required = ["iteration", "C_current", "gamma"]
            missing = [f for f in required if f not in data]
            if missing:
                errors.append(f"Champs manquants: {missing}")
        
        elif format_type == "prc_simulation_history":
            if "history" not in data:
                errors.append("Champ 'history' manquant")
        
        return {
            "valid": len(errors) == 0,
            "format": format_type,
            "version": version,
            "errors": errors if errors else None
        }
    
    except Exception as e:
        return {
            "valid": False,
            "format": "unknown",
            "version": "unknown",
            "errors": [str(e)]
        }


def get_format_info(filepath: Union[str, Path]) -> str:
    """
    Retourne une description lisible du format d'un fichier.
    
    Args:
        filepath: Chemin du fichier
    
    Returns:
        Chaîne descriptive du format
    """
    info = validate_format(filepath)
    
    if not info["valid"]:
        return f"INVALIDE: {info['errors']}"
    
    format_descriptions = {
        "prc_information_space": "Encodage InformationSpace (D)",
        "prc_simulation_state": "État de simulation (snapshot)",
        "prc_simulation_history": "Historique complet (trajectoire)",
        "prc_experiment_spec": "Spécification d'expérience"
    }
    
    desc = format_descriptions.get(info["format"], "Format inconnu")
    return f"{desc} (version {info['version']})"