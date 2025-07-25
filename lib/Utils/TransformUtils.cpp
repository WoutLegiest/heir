#include "lib/Utils/TransformUtils.h"

#include <set>
#include <string>
#include <string_view>

#include "mlir/include/mlir/Dialect/Arith/IR/Arith.h"    // from @llvm-project
#include "mlir/include/mlir/Dialect/Func/IR/FuncOps.h"   // from @llvm-project
#include "mlir/include/mlir/Dialect/MemRef/IR/MemRef.h"  // from @llvm-project
#include "mlir/include/mlir/IR/BuiltinOps.h"             // from @llvm-project
#include "mlir/include/mlir/IR/Visitors.h"               // from @llvm-project

namespace mlir {
namespace heir {

func::FuncOp detectEntryFunction(ModuleOp moduleOp,
                                 std::string_view entryFunction) {
  // get from user input
  auto entryFunc = moduleOp.lookupSymbol<func::FuncOp>(entryFunction);
  if (!entryFunc) {
    // detect the entry function with the following heuristic:
    // 1. the function name does not contain "__"
    // 2. the function is not a declaration
    // 3. the function is not called by any other function
    // 4. the first function that satisfies the above conditions

    // get all the called functions
    std::set<std::string> calledFuncs;
    moduleOp->walk<WalkOrder::PreOrder>([&](func::CallOp callOp) {
      auto callee = callOp.getCallee();
      calledFuncs.insert(std::string(callee));
    });

    moduleOp->walk<WalkOrder::PreOrder>([&](func::FuncOp funcOp) {
      auto funcSymName = funcOp.getSymName();
      if (funcSymName.find("__") != std::string::npos ||
          calledFuncs.find(std::string(funcSymName)) != calledFuncs.end() ||
          funcOp.isDeclaration()) {
        return WalkResult::advance();
      }
      entryFunc = funcOp;
      return WalkResult::interrupt();
    });
  }
  // still no result then emit warning
  if (!entryFunc) {
    moduleOp->emitWarning(
        "Entry function not found, please provide entry-function in the pass "
        "options");
  }
  return entryFunc;
}

Value convertIntegerValueToMemrefOfBits(Value integer, OpBuilder &b,
                                        Location loc) {
  IntegerType argType = mlir::cast<IntegerType>(integer.getType());
  int width = argType.getWidth();
  if (width == 1) {
    return integer;
  }

  auto allocOp =
      b.create<memref::AllocOp>(loc, MemRefType::get({width}, b.getI1Type()));
  for (int i = 0; i < width; i++) {
    // These arith ops correspond to extracting the i-th bit
    // from the input
    auto shiftAmount =
        b.create<arith::ConstantOp>(loc, argType, b.getIntegerAttr(argType, i));
    auto bitMask = b.create<arith::ConstantOp>(
        loc, argType, b.getIntegerAttr(argType, 1 << i));
    auto andOp = b.create<arith::AndIOp>(loc, integer, bitMask);
    auto shifted = b.create<arith::ShRSIOp>(loc, andOp, shiftAmount);
    b.create<memref::StoreOp>(
        loc, b.create<arith::TruncIOp>(loc, b.getI1Type(), shifted), allocOp,
        ValueRange{b.create<arith::ConstantIndexOp>(loc, i)});
  }

  return allocOp.getResult();
}

Value convertMemrefOfBitsToInteger(Value memref, Type resultType, OpBuilder &b,
                                   Location loc) {
  auto memrefType = cast<MemRefType>(memref.getType());
  auto integerType = cast<IntegerType>(resultType);
  assert(memrefType.getRank() == 1 && "Expected memref of bits to be 1D");

  Value result =
      b.create<arith::ConstantIntOp>(loc, integerType, 0).getResult();
  for (int i = 0; i < memrefType.getNumElements(); i++) {
    // The i-th bit of the memref is stored at bit position i
    auto loadOp = b.create<memref::LoadOp>(
        loc, memref, ValueRange{b.create<arith::ConstantIndexOp>(loc, i)});
    auto extOp = b.create<arith::ExtSIOp>(loc, integerType, loadOp.getResult());
    auto shiftAmount = b.create<arith::ConstantIntOp>(loc, integerType, i);
    auto shifted = b.create<arith::ShLIOp>(loc, extOp, shiftAmount);
    auto orOp = b.create<arith::OrIOp>(loc, integerType, result, shifted);
    result = orOp.getResult();
  }

  return result;
}

}  // namespace heir
}  // namespace mlir
