// Single sbox table lookup for AES encryption.
//
// RUN: heir-opt --mlir-to-cggi --scheme-to-tfhe-rs %s | heir-translate --emit-tfhe-rust --use-levels > %S/src/fn_under_test.rs
// RUN: cargo run --release --manifest-path %S/Cargo.toml --bin main_sbox | FileCheck %s

// CHECK: 637C777B

#map = affine_map<(d0) -> (d0)>
module {
  func.func @sub_bytes(%arg0: i8 {secret.secret}) -> i8 {
    %cst = arith.constant dense<"0x637C777BF26B6FC53001672BFED7AB76CA82C97DFA5947F0ADD4A2AF9CA472C0B7FD9326363FF7CC34A5E5F171D8311504C723C31896059A071280E2EB27B27509832C1A1B6E5AA0523BD6B329E32F8453D100ED20FCB15B6ACBBE394A4C58CFD0EFAAFB434D338545F9027F503C9FA851A3408F929D38F5BCB6DA2110FFF3D2CD0C13EC5F974417C4A77E3D645D197360814FDC222A908846EEB814DE5E0BDBE0323A0A4906245CC2D3AC629195E479E7C8376D8DD54EA96C56F4EA657AAE08BA78252E1CA6B4C6E8DD741F4BBD8B8A703EB5664803F60E613557B986C11D9EE1F8981169D98E949B1E87E9CE5528DF8CA1890DBFE6426841992D0FB054BB16"> : tensor<256xi8>
    %2 = arith.index_cast %arg0 : i8 to index
    %extracted = tensor.extract %cst[%2] : tensor<256xi8>
    return %extracted : i8
  }
}
