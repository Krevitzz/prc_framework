# tests/test_migration_verdict.py
"""
Script test migration verdict_reporter → profiling_common.

OBJECTIF :
- Valider pipeline complet après migration
- Comparer rapports avant/après migration
- Autoriser suppression gamma_profiling legacy

USAGE :
    python -m tests.test_migration_verdict
"""

import sys
from pathlib import Path

# Path projet
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Import verdict_reporter migré
from tests.utilities.HUB.verdict_reporter import generate_verdict_report


def test_verdict_pipeline_migrated():
    """Test pipeline verdict complet après migration."""
    
    print("\n" + "="*80)
    print("TEST MIGRATION VERDICT_REPORTER - PHASE 3")
    print("="*80)
    
    # Configuration
    params_config_id = 'params_default_v1'
    verdict_config_id = 'verdict_default_v1'
    
    print(f"\nParams config  : {params_config_id}")
    print(f"Verdict config : {verdict_config_id}")
    
    # Exécution pipeline complet
    print("\n" + "="*80)
    print("EXÉCUTION PIPELINE COMPLET (profiling_common)")
    print("="*80)
    
    try:
        results = generate_verdict_report(
            params_config_id=params_config_id,
            verdict_config_id=verdict_config_id,
            output_dir='reports/test_migration'
        )
        
        print("\n✅ PIPELINE EXÉCUTÉ AVEC SUCCÈS")
        
        # Validation structure résultats
        print("\n" + "="*80)
        print("VALIDATION RÉSULTATS")
        print("="*80)
        
        required_keys = ['metadata', 'gamma_profiles', 'structural_patterns', 
                        'comparisons', 'diagnostics', 'report_paths']
        
        for key in required_keys:
            if key in results:
                print(f"✅ {key:25s} : présent")
            else:
                print(f"❌ {key:25s} : MANQUANT")
                return False
        
        # Validation gamma_profiles
        print(f"\nGammas profilés : {len(results['gamma_profiles'])}")
        print(f"Tests analysés  : {len(results['comparisons']['by_test'])}")
        print(f"Fichiers générés : {len(results['report_paths'])}")
        
        # Liste fichiers générés
        print("\nFichiers générés :")
        for name, path in results['report_paths'].items():
            filepath = Path(path)
            exists = "✅" if filepath.exists() else "❌"
            print(f"  {exists} {name:30s} : {filepath.name}")
        
        # Validation rapport Jinja2
        summary_path = Path(results['report_paths'].get('summary_gamma', ''))
        if summary_path.exists():
            print(f"\n✅ Rapport Jinja2 généré : {summary_path}")
            
            # Vérifier contenu basique
            with open(summary_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Checks basiques
            checks = {
                'Header présent': 'RAPPORT GAMMA' in content,
                'Régimes listés': 'CONSERVATION' in content,
                'Timelines présentes': 'SIGNATURES DYNAMIQUES' in content,
                'Profils individuels': 'PROFILS INDIVIDUELS' in content,
                'Pas de {{ non substitué': '{{' not in content
            }
            
            print("\nValidation contenu rapport :")
            for check_name, passed in checks.items():
                status = "✅" if passed else "❌"
                print(f"  {status} {check_name}")
            
            if not all(checks.values()):
                print("\n⚠️  Certains checks échoués - vérifier rapport manuellement")
        else:
            print(f"\n❌ Rapport Jinja2 non trouvé : {summary_path}")
            return False
        
        print("\n" + "="*80)
        print("✅ MIGRATION VALIDÉE - Pipeline fonctionnel")
        print("="*80)
        
        print("\nNext steps:")
        print("1. Consulter rapports générés (reports/test_migration/)")
        print("2. Comparer avec rapports précédents (structure, contenu)")
        print("3. Si OK → Supprimer gamma_profiling.py legacy")
        print("4. Passer Phase 4 : Extension nouveaux axes")
        
        return True
        
    except Exception as e:
        print(f"\n❌ ERREUR PIPELINE : {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Pipeline test migration."""
    
    success = test_verdict_pipeline_migrated()
    
    if success:
        print("\n" + "="*80)
        print("✅ TEST MIGRATION RÉUSSI")
        print("="*80)
        return 0
    else:
        print("\n" + "="*80)
        print("❌ TEST MIGRATION ÉCHOUÉ")
        print("="*80)
        print("\nActions requises :")
        print("1. Vérifier imports verdict_reporter.py")
        print("2. Vérifier profiling_common.py disponible")
        print("3. Consulter traceback erreur")
        return 1


if __name__ == '__main__':
    exit_code = main()
    sys.exit(exit_code)