#ifndef LIB_DIALECT_ARITH_CONVERSIONS_ARITHTOCGGIQUART_TD_
#define LIB_DIALECT_ARITH_CONVERSIONS_ARITHTOCGGIQUART_TD_

include "mlir/Pass/PassBase.td"

def ArithToCGGIQuart : Pass<"arith-to-cggi-quart"> {
  let summary = "Lower `arith` to `cggi` dialect and divide each operation into smaller parts.";
  let dependentDialects = [
    "mlir::arith::ArithDialect",
    "mlir::memref::MemRefDialect",
    "mlir::tensor::TensorDialect",
    "mlir::heir::cggi::CGGIDialect",
  ];
}

#endif  // LIB_DIALECT_ARITH_CONVERSIONS_ARITHTOCGGIQUART_ARITHTOCGGIQUART_TD_
