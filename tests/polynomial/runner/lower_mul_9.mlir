// WARNING: this file is autogenerated. Do not edit manually, instead see
// tests/polynomial/runner/generate_test_cases.py

//-------------------------------------------------------
// entry and check_prefix are re-set per test execution
// DEFINE: %{entry} =
// DEFINE: %{check_prefix} =

// DEFINE: %{compile} = heir-opt %s --heir-polynomial-to-llvm
// DEFINE: %{run} = mlir-cpu-runner -e %{entry} -entry-point-result=void --shared-libs="%mlir_lib_dir/libmlir_c_runner_utils%shlibext,%mlir_runner_utils"
// DEFINE: %{check} = FileCheck %s --check-prefix=%{check_prefix}
//-------------------------------------------------------

func.func private @printMemrefI32(memref<*xi32>) attributes { llvm.emit_c_interface }

// REDEFINE: %{entry} = test_9
// REDEFINE: %{check_prefix} = CHECK_TEST_9
// RUN: %{compile} | %{run} | %{check}

#ideal_9 = #polynomial.polynomial<1 + x**3>
#ring_9 = #polynomial.ring<cmod=8, ideal=#ideal_9>
!poly_ty_9 = !polynomial.polynomial<#ring_9>

func.func @test_9() {
  %const0 = arith.constant 0 : index
  %0 = polynomial.constant <-4 + x**1> : !poly_ty_9
  %1 = polynomial.constant <-1 + 3x**1> : !poly_ty_9
  %2 = polynomial.mul(%0, %1) : !poly_ty_9


  %3 = polynomial.to_tensor %2 : !poly_ty_9 -> tensor<3xi3>
  %tensor = arith.extsi %3 : tensor<3xi3> to tensor<3xi32>

  %ref = bufferization.to_memref %tensor : memref<3xi32>
  %U = memref.cast %ref : memref<3xi32> to memref<*xi32>
  func.call @printMemrefI32(%U) : (memref<*xi32>) -> ()
  return
}
// expected_result: Poly(3*x**2 - 13*x + 4, x, domain='ZZ[8]')
// CHECK_TEST_9: [-4, 3, 3]
