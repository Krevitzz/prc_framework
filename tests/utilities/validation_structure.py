# tests/utilities/validation_structure.py
"""
Validation complète architecture PRC 5.4.

Vérifie conformité :
- Registres (base, manager, post_processors)
- Tests (structure, COMPUTATION_SPECS, applicability)
- Configs YAML (params_*.yaml)
- Discovery & applicability
- Test engine initialization

CRITIQUE : La moindre erreur arrête l'exécution.

Usage:
    python -m tests.utilities.validation_structure
    python -m tests.utilities.validation_structure --verbose
"""

import sys
import traceback
from pathlib import Path
from typing import List, Dict, Tuple
import argparse


class ValidationError(Exception):
    """Erreur de validation structure."""
    pass


class StructureValidator:
    """Validateur architecture PRC 5.4."""
    
    def __init__(self, verbose: bool = False):
        self.verbose = verbose
        self.errors: List[str] = []
        self.warnings: List[str] = []
    
    def log(self, message: str, level: str = "INFO"):
        """Log message selon niveau."""
        prefix = {
            "INFO": "  ℹ",
            "OK": "  ✓",
            "WARN": "  ⚠",
            "ERROR": "  ✗",
            "SECTION": "\n═══",
        }
        
        if level == "ERROR":
            self.errors.append(message)
        elif level == "WARN":
            self.warnings.append(message)
        
        if self.verbose or level in ["OK", "ERROR", "WARN", "SECTION"]:
            print(f"{prefix.get(level, '  ')} {message}")
    
    def validate_all(self) -> bool:
        """
        Exécute toutes validations.
        
        Returns:
            True si tout OK, False sinon
        """
        print("\n" + "="*80)
        print("VALIDATION STRUCTURE PRC 5.4")
        print("="*80)
        
        validations = [
            ("Registres", self.validate_registries),
            ("Tests", self.validate_tests),
            ("Configs YAML", self.validate_configs),
            ("Discovery", self.validate_discovery),
            ("Test Engine", self.validate_test_engine),
        ]
        
        for section_name, validation_func in validations:
            self.log(f"{section_name}", "SECTION")
            
            try:
                validation_func()
                self.log(f"{section_name} : OK", "OK")
            except ValidationError as e:
                self.log(f"{section_name} : ÉCHEC - {e}", "ERROR")
                self._print_summary()
                return False
            except Exception as e:
                self.log(f"{section_name} : ERREUR CRITIQUE", "ERROR")
                self.log(traceback.format_exc(), "ERROR")
                self._print_summary()
                return False
        
        self._print_summary()
        return len(self.errors) == 0
    
    # ========================================================================
    # VALIDATION REGISTRES
    # ========================================================================
    
    def validate_registries(self):
        """Valide architecture registres."""
        
        # 1. BaseRegistry existe
        try:
            from tests.utilities.registries.base_registry import BaseRegistry, register_function
            self.log("BaseRegistry importée", "OK")
        except ImportError as e:
            raise ValidationError(f"BaseRegistry non trouvée : {e}")
        
        # 2. RegistryManager existe et est singleton
        try:
            from tests.utilities.registries.registry_manager import RegistryManager
            
            manager1 = RegistryManager()
            manager2 = RegistryManager()
            
            if manager1 is not manager2:
                raise ValidationError("RegistryManager n'est pas un singleton")
            
            self.log("RegistryManager singleton", "OK")
        except ImportError as e:
            raise ValidationError(f"RegistryManager non trouvé : {e}")
        
        # 3. POST_PROCESSORS existe
        try:
            from tests.utilities.registries.post_processors import POST_PROCESSORS, get_post_processor
            
            if not POST_PROCESSORS:
                raise ValidationError("POST_PROCESSORS vide")
            
            # Test fonction get_post_processor
            try:
                func = get_post_processor('round_4')
                result = func(3.14159265)
                if not isinstance(result, float):
                    raise ValidationError("Post-processor doit retourner float")
            except KeyError:
                raise ValidationError("Post-processor 'round_4' manquant")
            
            self.log(f"POST_PROCESSORS : {len(POST_PROCESSORS)} fonctions", "OK")
        except ImportError as e:
            raise ValidationError(f"POST_PROCESSORS non trouvé : {e}")
        
        # 4. Charger registres disponibles
        manager = RegistryManager()
        
        if not manager.registries:
            raise ValidationError("Aucun registre chargé par RegistryManager")
        
        self.log(f"Registres chargés : {list(manager.registries.keys())}", "OK")
        
        # 5. Valider algebra_registry (minimum requis)
        if 'algebra' not in manager.registries:
            raise ValidationError("Registre 'algebra' manquant (requis)")
        
        algebra = manager.registries['algebra']
        
        # Vérifier fonctions requises
        required_functions = ['matrix_norm', 'matrix_asymmetry']
        algebra_functions = list(algebra._functions.keys())
        
        for func_name in required_functions:
            if func_name not in algebra_functions:
                raise ValidationError(
                    f"Fonction 'algebra.{func_name}' manquante (requise)"
                )
        
        self.log(f"algebra_registry : {algebra_functions}", "OK")
    
    # ========================================================================
    # VALIDATION TESTS
    # ========================================================================
    
    def validate_tests(self):
        """Valide structure tests."""
        
        tests_dir = Path("tests")
        test_files = list(tests_dir.glob("test_*.py"))
        
        if not test_files:
            raise ValidationError("Aucun fichier test_*.py trouvé")
        
        self.log(f"Fichiers test trouvés : {len(test_files)}", "INFO")
        
        # Importer et valider chaque test
        active_tests = []
        
        for test_file in test_files:
            # Skip deprecated
            if '_deprecated' in test_file.stem:
                self.log(f"Skip {test_file.name} (deprecated)", "INFO")
                continue
            
            module_name = f"tests.{test_file.stem}"
            
            try:
                import importlib
                module = importlib.import_module(module_name)
                
                # Valider structure
                self._validate_test_module(module, test_file.name)
                
                active_tests.append(module.TEST_ID)
                self.log(f"{module.TEST_ID} : valide", "OK")
            
            except Exception as e:
                raise ValidationError(
                    f"Test {test_file.name} invalide : {e}"
                )
        
        if not active_tests:
            raise ValidationError("Aucun test actif valide")
        
        self.log(f"Tests actifs validés : {active_tests}", "OK")
    
    def _validate_test_module(self, module, filename: str):
        """Valide structure d'un module test."""
        
        # Attributs requis
        required_attrs = [
            'TEST_ID',
            'TEST_CATEGORY',
            'TEST_VERSION',
            'APPLICABILITY_SPEC',
            'COMPUTATION_SPECS',
        ]
        
        for attr in required_attrs:
            if not hasattr(module, attr):
                raise ValidationError(
                    f"{filename} manque attribut requis : {attr}"
                )
        
        # Version 5.4 obligatoire
        if module.TEST_VERSION != "5.4":
            raise ValidationError(
                f"{filename} version invalide : {module.TEST_VERSION} (attendu 5.4)"
            )
        
        # TEST_ID format CAT-NNN
        import re
        if not re.match(r'^[A-Z]{3,4}-\d{3}$', module.TEST_ID):
            raise ValidationError(
                f"{filename} TEST_ID invalide : {module.TEST_ID} (format CAT-NNN)"
            )
        
        # COMPUTATION_SPECS non vide
        if not module.COMPUTATION_SPECS:
            raise ValidationError(
                f"{filename} COMPUTATION_SPECS vide"
            )
        
        # Entre 1 et 5 métriques
        num_metrics = len(module.COMPUTATION_SPECS)
        if not (1 <= num_metrics <= 5):
            raise ValidationError(
                f"{filename} nombre métriques invalide : {num_metrics} (attendu 1-5)"
            )
        
        # Valider chaque métrique
        for metric_name, spec in module.COMPUTATION_SPECS.items():
            self._validate_computation_spec(spec, filename, metric_name)
        
        # Pas de FORMULAS legacy
        if hasattr(module, 'FORMULAS'):
            raise ValidationError(
                f"{filename} contient FORMULAS (obsolète en 5.4)"
            )
    
    def _validate_computation_spec(self, spec: dict, filename: str, metric_name: str):
        """Valide une spécification de calcul."""
        
        # registry_key obligatoire
        if 'registry_key' not in spec:
            raise ValidationError(
                f"{filename}.{metric_name} manque 'registry_key'"
            )
        
        # Format registre.fonction
        if '.' not in spec['registry_key']:
            raise ValidationError(
                f"{filename}.{metric_name} registry_key invalide : {spec['registry_key']} "
                f"(format attendu: registre.fonction)"
            )
        
        # default_params obligatoire
        if 'default_params' not in spec:
            raise ValidationError(
                f"{filename}.{metric_name} manque 'default_params'"
            )
        
        # post_process doit être string (clé POST_PROCESSORS)
        if 'post_process' in spec:
            if not isinstance(spec['post_process'], str):
                raise ValidationError(
                    f"{filename}.{metric_name} post_process doit être string, "
                    f"pas {type(spec['post_process'])}"
                )
            
            # Vérifier que clé existe
            from tests.utilities.registries.post_processors import POST_PROCESSORS
            if spec['post_process'] not in POST_PROCESSORS:
                raise ValidationError(
                    f"{filename}.{metric_name} post_process '{spec['post_process']}' "
                    f"non trouvé dans POST_PROCESSORS"
                )
    
    # ========================================================================
    # VALIDATION CONFIGS YAML
    # ========================================================================
    
    def validate_configs(self):
        """Valide configs YAML params."""
        
        config_dir = Path("tests/config/global")
        
        if not config_dir.exists():
            raise ValidationError(f"Répertoire {config_dir} manquant")
        
        # params_default_v1.yaml obligatoire
        default_params = config_dir / "params_default_v1.yaml"
        
        if not default_params.exists():
            raise ValidationError(
                f"Config obligatoire manquante : {default_params}"
            )
        
        # Charger et valider
        import yaml
        
        with open(default_params) as f:
            config = yaml.safe_load(f)
        
        # Validation structure
        required_keys = ['version', 'config_id', 'description', 'common']
        for key in required_keys:
            if key not in config:
                raise ValidationError(
                    f"params_default_v1.yaml manque clé : {key}"
                )
        
        # Valider common params
        common = config['common']
        required_params = [
            'explosion_threshold',
            'stability_tolerance',
            'epsilon',
        ]
        
        for param in required_params:
            if param not in common:
                raise ValidationError(
                    f"params_default_v1.yaml manque param : {param}"
                )
        
        self.log("params_default_v1.yaml : valide", "OK")
        
        # Lister autres configs disponibles
        other_configs = [f.name for f in config_dir.glob("params_*.yaml")]
        if other_configs:
            self.log(f"Autres configs params : {other_configs}", "INFO")
    
    # ========================================================================
    # VALIDATION DISCOVERY
    # ========================================================================
    
    def validate_discovery(self):
        """Valide discovery fonctionne."""
        
        try:
            from tests.utilities.discovery import discover_active_tests
            
            tests = discover_active_tests()
            
            if not tests:
                raise ValidationError("discover_active_tests() retourne vide")
            
            self.log(f"Tests découverts : {list(tests.keys())}", "OK")
            
            # Vérifier que tous tests ont structure valide
            for test_id, module in tests.items():
                if not hasattr(module, 'TEST_ID'):
                    raise ValidationError(
                        f"Test découvert {test_id} n'a pas TEST_ID"
                    )
                
                if module.TEST_ID != test_id:
                    raise ValidationError(
                        f"Incohérence TEST_ID : {test_id} vs {module.TEST_ID}"
                    )
        
        except ImportError as e:
            raise ValidationError(f"Impossible d'importer discovery : {e}")
    
    # ========================================================================
    # VALIDATION TEST ENGINE
    # ========================================================================
    
    def validate_test_engine(self):
        """Valide TestEngine peut s'initialiser."""
        
        try:
            from tests.utilities.test_engine import TestEngine
            
            engine = TestEngine()
            
            # Vérifier attributs requis
            if not hasattr(engine, 'registry_manager'):
                raise ValidationError("TestEngine manque registry_manager")
            
            if not hasattr(engine, 'config_loader'):
                raise ValidationError("TestEngine manque config_loader")
            
            self.log("TestEngine initialisé", "OK")
            
            # Vérifier que RegistryManager a chargé registres
            if not engine.registry_manager.registries:
                raise ValidationError("RegistryManager n'a pas chargé de registres")
            
            self.log(f"Registres disponibles : {list(engine.registry_manager.registries.keys())}", "OK")
        
        except ImportError as e:
            raise ValidationError(f"Impossible d'importer TestEngine : {e}")
    
    # ========================================================================
    # RÉSUMÉ
    # ========================================================================
    
    def _print_summary(self):
        """Affiche résumé validation."""
        print("\n" + "="*80)
        print("RÉSUMÉ VALIDATION")
        print("="*80)
        
        if self.errors:
            print(f"\n✗ ÉCHEC : {len(self.errors)} erreur(s)")
            for error in self.errors:
                print(f"  - {error}")
        else:
            print("\n✓ SUCCÈS : Toutes validations passées")
        
        if self.warnings:
            print(f"\n⚠ Avertissements : {len(self.warnings)}")
            for warning in self.warnings:
                print(f"  - {warning}")
        
        print("\n" + "="*80 + "\n")


def main():
    """Point d'entrée."""
    parser = argparse.ArgumentParser(
        description="Validation structure PRC 5.4"
    )
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Mode verbeux'
    )
    
    args = parser.parse_args()
    
    validator = StructureValidator(verbose=args.verbose)
    success = validator.validate_all()
    
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()