"""
test_registries_lite.py

Tests : prc.featuring.registries (matrix_2d, tensor_3d)

Méthodologie : Observations pures, validation features
"""

from pathlib import Path
import sys
import numpy as np

# Ajouter prc/ au path
sys.path.insert(0, str(Path(__file__).parent.parent))

from featuring.registries.matrix_2d_lite import (
    compute_trace,
    compute_eigenvalue_max,
    compute_asymmetry_norm,
    compute_condition_number,
    compute_determinant,
)
from featuring.registries.tensor_3d_lite import (
    compute_mode_variance_0,
    compute_mode_variance_1,
    compute_mode_variance_2,
)


# =============================================================================
# TESTS MATRIX 2D
# =============================================================================

def test_compute_trace():
    """Test trace matrice."""
    # Matrice identité
    state = np.eye(5)
    trace = compute_trace(state)
    
    return {
        'test': 'compute_trace',
        'trace_identity': trace,
        'correct_trace': abs(trace - 5.0) < 1e-10,
    }


def test_compute_eigenvalue_max():
    """Test eigenvalue_max."""
    # Matrice diagonale
    state = np.diag([1.0, 2.0, 3.0])
    eig_max = compute_eigenvalue_max(state)
    
    return {
        'test': 'compute_eigenvalue_max',
        'eig_max': eig_max,
        'correct_eig_max': abs(eig_max - 3.0) < 1e-10,
    }


def test_compute_asymmetry_norm():
    """Test asymmetry_norm."""
    # Matrice symétrique
    sym = np.array([[1, 2], [2, 3]])
    asym_sym = compute_asymmetry_norm(sym)
    
    # Matrice asymétrique
    asym = np.array([[1, 2], [3, 4]])
    asym_asym = compute_asymmetry_norm(asym)
    
    return {
        'test': 'compute_asymmetry_norm',
        'asym_symmetric': asym_sym,
        'asym_asymmetric': asym_asym,
        'symmetric_is_zero': abs(asym_sym) < 1e-10,
        'asymmetric_nonzero': asym_asym > 0,
    }


def test_compute_condition_number():
    """Test condition_number."""
    # Matrice bien conditionnée (identité)
    well_conditioned = np.eye(3)
    cond_good = compute_condition_number(well_conditioned)
    
    # Matrice mal conditionnée
    ill_conditioned = np.array([[1, 0.999], [0.999, 1]])
    cond_bad = compute_condition_number(ill_conditioned)
    
    return {
        'test': 'compute_condition_number',
        'cond_identity': cond_good,
        'cond_ill': cond_bad,
        'identity_is_one': abs(cond_good - 1.0) < 1e-5,
        'ill_is_large': cond_bad > 10,
    }


def test_compute_determinant():
    """Test determinant."""
    # Matrice identité
    state = np.eye(4)
    det = compute_determinant(state)
    
    return {
        'test': 'compute_determinant',
        'det_identity': det,
        'correct_det': abs(det - 1.0) < 1e-10,
    }


def test_matrix_2d_rank_validation():
    """Test validation rank pour matrix_2d."""
    # Rank 3 → ValueError
    tensor_3d = np.random.randn(3, 3, 3)
    
    try:
        compute_trace(tensor_3d)
        error_raised = False
    except ValueError:
        error_raised = True
    
    return {
        'test': 'matrix_2d_rank_validation',
        'error_raised_rank3': error_raised,
        'correct_validation': error_raised,
    }


# =============================================================================
# TESTS TENSOR 3D
# =============================================================================

def test_compute_mode_variance_0():
    """Test mode_variance_0."""
    # Tenseur uniforme → variance 0
    uniform = np.ones((3, 4, 5))
    var0 = compute_mode_variance_0(uniform)
    
    # Tenseur variable
    variable = np.random.randn(3, 4, 5)
    var0_variable = compute_mode_variance_0(variable)
    
    return {
        'test': 'compute_mode_variance_0',
        'var_uniform': var0,
        'var_variable': var0_variable,
        'uniform_is_zero': abs(var0) < 1e-10,
        'variable_nonzero': var0_variable > 0,
    }


def test_compute_mode_variance_1():
    """Test mode_variance_1."""
    variable = np.random.randn(3, 4, 5)
    var1 = compute_mode_variance_1(variable)
    
    return {
        'test': 'compute_mode_variance_1',
        'var1': var1,
        'var1_positive': var1 >= 0,
    }


def test_compute_mode_variance_2():
    """Test mode_variance_2."""
    variable = np.random.randn(3, 4, 5)
    var2 = compute_mode_variance_2(variable)
    
    return {
        'test': 'compute_mode_variance_2',
        'var2': var2,
        'var2_positive': var2 >= 0,
    }


def test_tensor_3d_rank_validation():
    """Test validation rank pour tensor_3d."""
    # Rank 2 → ValueError
    matrix = np.random.randn(3, 3)
    
    try:
        compute_mode_variance_0(matrix)
        error_raised = False
    except ValueError:
        error_raised = True
    
    return {
        'test': 'tensor_3d_rank_validation',
        'error_raised_rank2': error_raised,
        'correct_validation': error_raised,
    }


# =============================================================================
# RUNNER PRINCIPAL
# =============================================================================

def run_all_tests():
    """Execute tous les tests et affiche résultats."""
    tests = [
        # Matrix 2D
        test_compute_trace,
        test_compute_eigenvalue_max,
        test_compute_asymmetry_norm,
        test_compute_condition_number,
        test_compute_determinant,
        test_matrix_2d_rank_validation,
        # Tensor 3D
        test_compute_mode_variance_0,
        test_compute_mode_variance_1,
        test_compute_mode_variance_2,
        test_tensor_3d_rank_validation,
    ]
    
    results = []
    
    for test_fn in tests:
        try:
            result = test_fn()
            result['status'] = 'OK'
        except Exception as e:
            result = {
                'test': test_fn.__name__,
                'status': 'ERROR',
                'error': str(e),
            }
        
        results.append(result)
    
    return results


if __name__ == '__main__':
    print("=== TESTS registries_lite ===\n")
    
    results = run_all_tests()
    
    for result in results:
        test_name = result.get('test', 'unknown')
        status = result.get('status', 'UNKNOWN')
        
        print(f"[{status}] {test_name}")
        
        if status == 'ERROR':
            print(f"  ERROR: {result.get('error')}")
        else:
            for key, value in result.items():
                if key not in ['test', 'status']:
                    print(f"  {key}: {value}")
        
        print()
    
    # Résumé
    n_ok = sum(1 for r in results if r.get('status') == 'OK')
    n_error = sum(1 for r in results if r.get('status') == 'ERROR')
    
    print(f"=== RÉSUMÉ : {n_ok} OK, {n_error} ERROR ===")
