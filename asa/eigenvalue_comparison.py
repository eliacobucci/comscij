#!/usr/bin/env python3
"""
Eigenvalue Algorithm Comparison: Galileo v57.FOR vs Python numpy.linalg.eigh

This script compares the Fortran Power Method algorithm from the original Galileo 
v57.FOR source code with our current Python numpy.linalg.eigh implementation
to ensure mathematical equivalence.

Copyright (c) 2025 Joseph Woelfel and Emary Iacobucci. All rights reserved.
"""

import numpy as np
import time
from typing import Tuple, List
from huey_gpu_conversational_experiment import HueyGPUConversationalNetwork


def fortran_power_method_algorithm(matrix: np.ndarray, tolerance: float = 0.0001, 
                                   max_iterations: int = 500) -> Tuple[np.ndarray, np.ndarray]:
    """
    Python implementation of the Galileo v57.FOR Power Method eigenvalue algorithm.
    
    This replicates the exact algorithm from lines 1706-1849 in v57.FOR:
    1. Matrix exponentiation (quinque = 5th power)
    2. Power iteration for each eigenvalue/eigenvector
    3. Deflation after each eigenvalue is found
    4. Sorting in descending order
    
    Args:
        matrix: Input association matrix (symmetric)
        tolerance: Convergence tolerance (default 0.0001)
        max_iterations: Maximum iterations per eigenvalue (default 500)
        
    Returns:
        Tuple of (eigenvalues, eigenvectors) sorted in descending order
    """
    msize = matrix.shape[0]
    eigenvals = np.zeros(msize)
    eigenvecs = np.zeros((msize, msize))
    iterations_per_root = np.zeros(msize, dtype=int)
    
    # Copy original matrix for deflation
    smean = matrix.copy().astype(np.float64)
    
    print(f"🔄 Fortran Power Method: Processing {msize}x{msize} matrix")
    
    # Process each eigenvalue
    for kount in range(msize):
        print(f"   Finding eigenvalue {kount + 1}/{msize}...")
        
        # QUINQUE-EXPONENTIATE THE MATRIX (5th power)
        # Lines 1706-1725 in v57.FOR
        z = smean.copy()
        for j in range(4):  # Raise to 5th power total
            y = np.zeros((msize, msize))
            for k in range(msize):
                for l in range(msize):
                    for m in range(msize):
                        y[k, l] += z[k, m] * smean[m, l]
            z = y.copy()
        
        # ITERATE TO EQUIVALENCE PRODUCT VECTORS
        # Lines 1727-1777 in v57.FOR (Power Method)
        vt = np.ones(msize)  # Initial vector
        iterations = 1
        
        while iterations <= max_iterations:
            # Matrix-vector multiply: C = Z * VT
            c = np.zeros(msize)
            for m in range(msize):
                for n in range(msize):
                    c[m] += z[m, n] * vt[n]
            
            # Find dominant element for normalization
            div = c[0]
            for j in range(1, msize):
                if abs(c[j]) > abs(div):
                    div = c[j]
            
            if div == 0:
                print(f"   Warning: Zero eigenvalue at position {kount + 1}")
                break
            
            # Check convergence
            converged = True
            for k in range(msize):
                ctmp = c[k] / div
                if abs(ctmp - vt[k]) > tolerance:
                    converged = False
                    break
                vt[k] = ctmp
            
            if converged:
                break
                
            iterations += 1
        
        iterations_per_root[kount] = iterations
        
        # RETRIEVE ROOT AND NORMALIZE VECTOR
        # Lines 1778-1790 in v57.FOR
        div = np.sign(div) * (abs(div) ** 0.2)  # 5th root with sign preservation
        eigenvals[kount] = div
        
        # Normalize eigenvector
        sumc = np.sum(vt ** 2)
        p = np.sqrt(sumc)
        q = np.sqrt(abs(div))  # Take square root of eigenvalue magnitude
        
        if p > 0:
            eigenvecs[:, kount] = (vt / p) * q
        else:
            eigenvecs[:, kount] = vt
        
        # COMPUTE RESIDUAL MATRIX (Deflation)
        # Lines 1795-1802 in v57.FOR
        sgn = -1 if div > 0 else 1
        for k in range(msize):
            for l in range(msize):
                smean[k, l] += sgn * (eigenvecs[k, kount] * eigenvecs[l, kount])
    
    # SORT VECTORS FROM NATURAL TO DESCENDING ORDER
    # Lines 1804-1822 in v57.FOR
    sorted_indices = np.argsort(eigenvals)[::-1]  # Descending order
    eigenvals = eigenvals[sorted_indices]
    eigenvecs = eigenvecs[:, sorted_indices]
    
    print(f"   ✅ Power Method complete: {msize} eigenvalues found")
    print(f"   Iterations per root: {iterations_per_root}")
    
    return eigenvals, eigenvecs


def numpy_eigenvalue_algorithm(matrix: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Python numpy.linalg.eigh eigenvalue algorithm (current implementation).
    
    This is the algorithm currently used in our Huey GPU system.
    """
    print(f"🚀 NumPy eigh: Processing {matrix.shape[0]}x{matrix.shape[0]} matrix")
    
    start_time = time.perf_counter()
    eigenvals, eigenvecs = np.linalg.eigh(matrix)
    
    # Sort in descending order to match Fortran
    idx = np.argsort(eigenvals)[::-1]
    eigenvals = eigenvals[idx]
    eigenvecs = eigenvecs[:, idx]
    
    elapsed_time = time.perf_counter() - start_time
    print(f"   ✅ NumPy eigh complete in {elapsed_time:.4f}s")
    
    return eigenvals, eigenvecs


def create_test_matrix(size: int = 5, matrix_type: str = "symmetric") -> np.ndarray:
    """Create a test matrix for comparison."""
    np.random.seed(42)  # Reproducible results
    
    if matrix_type == "symmetric":
        # Create positive definite symmetric matrix
        A = np.random.randn(size, size)
        matrix = A @ A.T + np.eye(size) * 0.1  # Add small diagonal term
    elif matrix_type == "association":
        # Create association matrix similar to Huey output
        matrix = np.random.rand(size, size)
        matrix = (matrix + matrix.T) / 2  # Make symmetric
        np.fill_diagonal(matrix, 1.0)  # Self-associations = 1
    else:
        # Simple test matrix
        matrix = np.array([
            [4, 2, 1],
            [2, 3, 0.5],
            [1, 0.5, 2]
        ], dtype=np.float64)
        if size != 3:
            # Extend to requested size
            new_matrix = np.eye(size) * 2
            min_dim = min(3, size)
            new_matrix[:min_dim, :min_dim] = matrix[:min_dim, :min_dim]
            matrix = new_matrix
    
    return matrix


def compare_algorithms() -> None:
    """Compare both eigenvalue algorithms with various test matrices."""
    print("🧪 EIGENVALUE ALGORITHM COMPARISON")
    print("=" * 50)
    print("Comparing Galileo v57.FOR Power Method vs NumPy eigh")
    print()
    
    test_cases = [
        ("Small symmetric (3x3)", 3, "simple"),
        ("Medium symmetric (5x5)", 5, "symmetric"),
        ("Association matrix (5x5)", 5, "association"),
        ("Larger matrix (8x8)", 8, "symmetric"),
    ]
    
    all_results = []
    
    for case_name, size, matrix_type in test_cases:
        print(f"📊 TEST CASE: {case_name}")
        print("-" * 40)
        
        # Create test matrix
        test_matrix = create_test_matrix(size, matrix_type)
        print(f"Test matrix ({size}x{size}):")
        print(test_matrix)
        print()
        
        try:
            # Run Fortran Power Method algorithm
            print("🔄 Running Fortran Power Method...")
            fortran_start = time.perf_counter()
            fortran_eigenvals, fortran_eigenvecs = fortran_power_method_algorithm(test_matrix)
            fortran_time = time.perf_counter() - fortran_start
            
            # Run NumPy algorithm
            print("🚀 Running NumPy eigh...")
            numpy_start = time.perf_counter()
            numpy_eigenvals, numpy_eigenvecs = numpy_eigenvalue_algorithm(test_matrix)
            numpy_time = time.perf_counter() - numpy_start
            
            # Compare results
            print("\n📈 COMPARISON RESULTS:")
            print(f"{'Method':<20} {'Time (s)':<12} {'Eigenvalues'}")
            print("-" * 60)
            print(f"{'Fortran Power':<20} {fortran_time:<12.6f} {fortran_eigenvals}")
            print(f"{'NumPy eigh':<20} {numpy_time:<12.6f} {numpy_eigenvals}")
            
            # Calculate differences
            eigenval_diff = np.abs(fortran_eigenvals - numpy_eigenvals)
            max_eigenval_diff = np.max(eigenval_diff)
            
            print(f"\n📊 ACCURACY ANALYSIS:")
            print(f"   Max eigenvalue difference: {max_eigenval_diff:.8f}")
            print(f"   Eigenvalue differences: {eigenval_diff}")
            
            # Check eigenvector alignment (may differ by sign/phase)
            eigenvec_similarities = []
            for i in range(size):
                # Compare normalized eigenvectors (handle sign ambiguity)
                v1 = fortran_eigenvecs[:, i] / np.linalg.norm(fortran_eigenvecs[:, i])
                v2 = numpy_eigenvecs[:, i] / np.linalg.norm(numpy_eigenvecs[:, i])
                similarity = max(abs(np.dot(v1, v2)), abs(np.dot(v1, -v2)))
                eigenvec_similarities.append(similarity)
            
            min_similarity = min(eigenvec_similarities)
            print(f"   Min eigenvector similarity: {min_similarity:.8f}")
            
            # Determine if algorithms are equivalent
            eigenval_tolerance = 1e-6
            eigenvec_tolerance = 0.999
            
            eigenvals_match = max_eigenval_diff < eigenval_tolerance
            eigenvecs_match = min_similarity > eigenvec_tolerance
            
            status = "✅ EQUIVALENT" if (eigenvals_match and eigenvecs_match) else "❌ DIFFERENT"
            print(f"   Algorithm equivalence: {status}")
            
            all_results.append({
                'case': case_name,
                'size': size,
                'eigenval_diff': max_eigenval_diff,
                'eigenvec_similarity': min_similarity,
                'fortran_time': fortran_time,
                'numpy_time': numpy_time,
                'equivalent': eigenvals_match and eigenvecs_match
            })
            
        except Exception as e:
            print(f"❌ Error in test case: {e}")
            all_results.append({
                'case': case_name,
                'size': size,
                'error': str(e)
            })
        
        print("\n" + "=" * 50 + "\n")
    
    # Summary
    print("📋 OVERALL SUMMARY")
    print("-" * 30)
    
    successful_tests = [r for r in all_results if 'error' not in r]
    if successful_tests:
        avg_eigenval_diff = np.mean([r['eigenval_diff'] for r in successful_tests])
        avg_eigenvec_similarity = np.mean([r['eigenvec_similarity'] for r in successful_tests])
        all_equivalent = all(r['equivalent'] for r in successful_tests)
        
        print(f"Tests completed: {len(successful_tests)}/{len(all_results)}")
        print(f"Average eigenvalue difference: {avg_eigenval_diff:.8f}")
        print(f"Average eigenvector similarity: {avg_eigenvec_similarity:.8f}")
        print(f"All tests equivalent: {'✅ YES' if all_equivalent else '❌ NO'}")
        
        if all_equivalent:
            print("\n🎉 CONCLUSION: The algorithms are mathematically equivalent!")
            print("   Our Python numpy.linalg.eigh implementation produces the same")
            print("   results as the original Galileo Fortran Power Method algorithm.")
        else:
            print("\n⚠️  CONCLUSION: Algorithms show differences.")
            print("   Further investigation may be needed to understand discrepancies.")
    
    return all_results


if __name__ == "__main__":
    # Run the comprehensive comparison
    results = compare_algorithms()
    
    print(f"\n🔬 EIGENVALUE COMPARISON COMPLETE")
    print(f"Results available for further analysis if needed.")