// RUN: heir-opt --color %s > %t
// RUN: FileCheck %s < %t

// This simply tests for syntax.

#encoding = #lwe.polynomial_evaluation_encoding<cleartext_start=30, cleartext_bitwidth=3>

#my_poly = #polynomial.polynomial<1 + x**1024>
// cmod is 64153 * 2521
#ring1 = #polynomial.ring<cmod=161729713, ideal=#my_poly>
#ring2 = #polynomial.ring<cmod=2521, ideal=#my_poly>

#params = #lwe.rlwe_params<dimension=2, ring=#ring1>
#params1 = #lwe.rlwe_params<dimension=3, ring=#ring1>
#params2 = #lwe.rlwe_params<dimension=2, ring=#ring2>

!pt = !lwe.rlwe_plaintext<encoding=#encoding, ring=#ring1>

!ct = !lwe.rlwe_ciphertext<encoding=#encoding, rlwe_params=#params>
!ct1 = !lwe.rlwe_ciphertext<encoding=#encoding, rlwe_params=#params1>
!ct2 = !lwe.rlwe_ciphertext<encoding=#encoding, rlwe_params=#params2>

// CHECK: module
module {
  func.func @test_multiply(%arg0 : !ct, %arg1: !ct) -> !ct {
    %add = bgv.add(%arg0, %arg1) : !ct
    %sub = bgv.sub(%arg0, %arg1) : !ct
    %neg = bgv.negate(%arg0) : !ct

    %0 = bgv.mul(%arg0, %arg1) : !ct -> !ct1
    %1 = bgv.relinearize(%0) {from_basis = array<i32: 0, 1, 2>, to_basis = array<i32: 0, 1> } : (!ct1) -> !ct
    %2 = bgv.modulus_switch(%1) {to_ring = #ring2} : (!ct) -> !ct2
    // CHECK: rlwe_params = <dimension = 3, ring = <cmod=161729713, ideal=#polynomial.polynomial<1 + x**1024>>>
    return %arg0 : !ct
  }

  func.func @test_ciphertext_plaintext(%arg0: !pt, %arg1: !pt, %arg2: !pt, %arg3: !ct) -> !ct {
    %add = bgv.add_plain(%arg3, %arg0) : !ct
    %sub = bgv.sub_plain(%add, %arg1) : !ct
    %mul = bgv.mul_plain(%sub, %arg2) : !ct
    // CHECK: rlwe_params = <dimension = 3, ring = <cmod=161729713, ideal=#polynomial.polynomial<1 + x**1024>>>
    return %mul : !ct
  }
}
