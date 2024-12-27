#encoding = #lwe.unspecified_bit_field_encoding<cleartext_bitwidth = 32>
!ct_ty = !lwe.lwe_ciphertext<encoding = #encoding>


module attributes {tf_saved_model.semantics} {
  func.func @test_affine(%arg0: memref<1x1x!ct_ty>) -> memref<1x1x!ct_ty> {
    %0 = cggi.create_trivial  {value = 429 : i32} : () -> !ct_ty
    %1 = cggi.create_trivial  {value = 33 : i32} : () -> !ct_ty
    %2 = affine.load %arg0[0, 0] : memref<1x1x!ct_ty>
    %c0 = arith.constant 0 : index
    %alloc = memref.alloc() {alignment = 64 : i64} : memref<1x1x!ct_ty>
    %3 = cggi.mul %2, %1 : !ct_ty
    %4 = cggi.add %3, %0 : !ct_ty
    affine.store %4, %alloc[0, 0] : memref<1x1x!ct_ty>
    return %alloc : memref<1x1x!ct_ty>
  }
}
