# tests/utilities/config_loader.py
"""
Utilitaire centralisé chargement configs YAML.

Architecture Charter 5.4 - Section 12.6
"""

import yaml
from pathlib import Path
from typing import Dict, Any, Optional
import warnings

class ConfigLoader:
    """
    Gestionnaire centralisé configs YAML.
    
    Responsabilités :
    1. Charger n'importe quel type config (params, scoring, thresholds)
    2. Gérer fusion global + specific automatique
    3. Cache pour performance
    4. Validation basique structure
    """
    
    BASE_PATH = Path("tests/config")
    
    def __init__(self):
        self._cache: Dict[str, Dict] = {}
    
    def load(
        self,
        config_type: str,
        config_id: str,
        test_id: Optional[str] = None,
        force_reload: bool = False
    ) -> Dict[str, Any]:
        """
        Charge config avec fusion auto global + specific.
        
        Args:
            config_type: 'params' | 'verdict' 
            config_id: Ex 'params_default_v1', 'scoring_conservative_v1'
            test_id: Ex 'UNIV-001' (optionnel, pour override)
            force_reload: Ignorer cache
        
        Returns:
            dict config fusionné
        
        Examples:
            >>> loader = ConfigLoader()
            >>> params = loader.load('params', 'params_default_v1')
            >>> params_univ = loader.load('params', 'params_default_v1', 'UNIV-001')
        
        Raises:
            FileNotFoundError: Si config global absent
            ValueError: Si type config invalide
        """
        # Validation
        valid_types = ['params', 'verdict']
        if config_type not in valid_types:
            raise ValueError(
                f"Type config invalide '{config_type}'. "
                f"Attendu: {valid_types}"
            )
        
        # Clé cache
        cache_key = f"{config_type}:{config_id}:{test_id or 'global'}"
        
        if not force_reload and cache_key in self._cache:
            return self._cache[cache_key].copy()
        
        # 1. Charger global (obligatoire)
        global_config = self._load_global(config_type, config_id)
        
        # 2. Charger specific (optionnel)
        if test_id:
            specific_config = self._load_specific(config_type, test_id)
            if specific_config:
                merged = self._merge_configs(global_config, specific_config)
            else:
                merged = global_config
        else:
            merged = global_config
        
        # Cache et retour
        self._cache[cache_key] = merged.copy()
        return merged
    
    def _load_global(self, config_type: str, config_id: str) -> Dict:
        """Charge config global."""
        path = self.BASE_PATH / "global" / f"{config_id}.yaml"
        
        if not path.exists():
            raise FileNotFoundError(
                f"Config global non trouvée: {path}\n"
                f"Vérifier que {config_id}.yaml existe dans tests/config/global/"
            )
        
        with open(path, 'r') as f:
            config = yaml.safe_load(f)
        
        # Validation basique
        self._validate_config(config, config_type, config_id)
        
        return config
    
    def _load_specific(self, config_type: str, test_id: str) -> Optional[Dict]:
        """Charge config spécifique test (si existe)."""
        test_dir = self.BASE_PATH / "tests" / test_id
        
        if not test_dir.exists():
            return None
        
        # Chercher fichiers matching (ex: params_custom_v1.yaml)
        pattern = f"{config_type}_*.yaml"
        matching_files = list(test_dir.glob(pattern))
        
        if not matching_files:
            return None
        
        # Si plusieurs, prendre le plus récent (ou alphabétique)
        selected = sorted(matching_files)[-1]
        
        with open(selected, 'r') as f:
            config = yaml.safe_load(f)
        
        return config
    
    def _merge_configs(self, global_config: Dict, specific_config: Dict) -> Dict:
        """
        Fusionne configs (specific override global).
        
        Stratégie fusion :
        - Clés top-level : specific écrase global
        - Dicts imbriqués : merge récursif
        - Listes : specific remplace global
        """
        merged = global_config.copy()
        
        for key, value in specific_config.items():
            if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
                # Merge récursif dicts
                merged[key] = self._merge_dicts(merged[key], value)
            else:
                # Override direct
                merged[key] = value
        
        return merged
    
    def _merge_dicts(self, base: Dict, override: Dict) -> Dict:
        """Merge récursif dictionnaires."""
        result = base.copy()
        
        for key, value in override.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = self._merge_dicts(result[key], value)
            else:
                result[key] = value
        
        return result
    
    def _validate_config(self, config: Dict, config_type: str, config_id: str):
        """Validation basique structure."""
        required_keys = ['version', 'config_id', 'description']
        
        missing = [k for k in required_keys if k not in config]
        if missing:
            warnings.warn(
                f"Config {config_id} manque clés: {missing}\n"
                f"Recommandé: ajouter metadata (version, description, etc.)"
            )
        
        # Vérifier cohérence ID
        if 'config_id' in config and config['config_id'] != config_id:
            warnings.warn(
                f"ID incohérent: fichier={config_id}, contenu={config['config_id']}"
            )
    
    def list_available(self, config_type: str) -> Dict[str, list]:
        """
        Liste configs disponibles.
        
        Returns:
            {
                'global': ['params_default_v1', ...],
                'tests': {
                    'UNIV-001': ['params_custom_v1', ...],
                    ...
                }
            }
        """
        result = {'global': [], 'tests': {}}
        
        # Configs globales
        global_dir = self.BASE_PATH / "global"
        if global_dir.exists():
            pattern = f"{config_type}_*.yaml"
            result['global'] = [
                f.stem for f in global_dir.glob(pattern)
            ]
        
        # Configs spécifiques
        tests_dir = self.BASE_PATH / "tests"
        if tests_dir.exists():
            for test_dir in tests_dir.iterdir():
                if test_dir.is_dir():
                    pattern = f"{config_type}_*.yaml"
                    matching = [f.stem for f in test_dir.glob(pattern)]
                    if matching:
                        result['tests'][test_dir.name] = matching
        
        return result
    
    def clear_cache(self):
        """Vide cache (utile après modification configs)."""
        self._cache.clear()


# Instance singleton
_loader = None

def get_loader() -> ConfigLoader:
    """Récupère instance singleton ConfigLoader."""
    global _loader
    if _loader is None:
        _loader = ConfigLoader()
    return _loader
