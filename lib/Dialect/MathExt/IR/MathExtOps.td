#ifndef LIB_DIALECT_MATHEXT_IR_MATHEXTOPS_TD_
#define LIB_DIALECT_MATHEXT_IR_MATHEXTOPS_TD_

include "lib/Dialect/MathExt/IR/MathExtDialect.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/Interfaces/InferTypeOpInterface.td"
include "mlir/IR/BuiltinTypes.td"
include "mlir/IR/OpBase.td"

class MathExt_Op<string mnemonic, list<Trait> traits = []> :
        Op<MathExt_Dialect, mnemonic, traits> {
  let cppNamespace = "::mlir::heir::math_ext";
}

def MathExt_SignOp : MathExt_Op<"sign", [Pure, ElementwiseMappable,SameOperandsAndResultType]> {
  let summary = "Returns the sign of the input value";
  let description = [{
    Returns -1 if the input is negative, 0 if it is zero, and 1 if it is positive.
    The behavior is undefined for NaN inputs.
  }];
  let arguments = (ins SignlessIntegerOrFloatLike:$value);
  let results = (outs SignlessIntegerOrFloatLike:$result);
   let assemblyFormat = "$value attr-dict `:` type($result)";
}

def MathExt_OneHotOp : MathExt_Op<"one_hot", [Pure]> {
  let summary = "Returns the one-hot encoding of the input value";
  let description = [{
   Return a tensor of boolean values where only the index corresponding to the input value is true.
  }];
  let arguments = (ins Builtin_Integer:$value, Optional<MemRefOf<[I1]>>:$memref_result);
  let results = (outs AnyTypeOf<[TensorOf<[I1]>, MemRefOf<[I1]>]>:$result);
  // let assemblyFormat = "$value (`into` $memref_result^)? `:` type($value) (`, ` type($memref_result)^)? `->` type($result) attr-dict ";

  let assemblyFormat = "$value (`into` $memref_result^)? attr-dict `:` functional-type(operands, results)";
    }

def MathExt_MemOneHotOp : MathExt_Op<"one_hot_mem", [Pure, MemoryEffects<[MemWrite]>]> {
  let summary = "Returns the one-hot encoding of the input value";
  let description = [{
   Return a memref of boolean values where only the index corresponding to the input value is true.
  }];
  let arguments = (ins Builtin_Integer:$value, Arg<AnyMemRef, "the reference to store to",
                           [MemWrite]>:$memref);
  // let assemblyFormat = "$value attr-dict `:` type($value) `->` type($result)";
}



#endif  // LIB_DIALECT_MATHEXT_IR_MATHEXTOPS_TD_
