// RUN: heir-opt --arith-quarter-wide-int  %s | FileCheck %s

// CHECK-LABEL: func @test_simple_split
// CHCK-COUNT-9: arith.muli
// CHCK-COUNT-7: arith.addi
// CHCK-COUNT-3: arith.shrui
// CHCK-COUNT-3: arith.addi
func.func @test_simple_split(%arg0: i32, %arg1: i32) -> i32 {
  %1 = arith.constant 522067228: i32 // Hex 1f1e1d1c
  %2 = arith.constant 31 : i8
  // %c0 = arith.constant 0 : index
  %3 = arith.extui %2 : i8 to i32
  // %a4 = tensor.extract %arg1[%c0] : tensor<4xi32>
  // %4 = arith.extui %arg0 : i16 to i32
  // %5 = arith.muli %1, %a4 : i32
  %6 = arith.addi %arg1, %3 : i32
  %7 = arith.addi %arg0, %6 : i32
  return %6 : i32
}
