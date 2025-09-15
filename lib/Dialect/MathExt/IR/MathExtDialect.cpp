#include "lib/Dialect/MathExt/IR/MathExtDialect.h"

#include "lib/Dialect/MathExt/IR/MathExtOps.h"
#include "mlir/include/mlir/Dialect/Arith/IR/Arith.h"  // from @llvm-project
#include "mlir/include/mlir/Dialect/Bufferization/IR/BufferizableOpInterface.h"  // from @llvm-project
#include "mlir/include/mlir/Dialect/Linalg/IR/Linalg.h"  // Needed for linalg::FillOp
#include "mlir/include/mlir/Dialect/MemRef/IR/MemRef.h"  // from @llvm-project
#include "mlir/include/mlir/IR/Builders.h"               // from @llvm-project
#include "mlir/include/mlir/IR/BuiltinTypes.h"           // from @llvm-project
#include "mlir/include/mlir/IR/DialectImplementation.h"  // from @llvm-project
#include "mlir/include/mlir/IR/OpDefinition.h"           // from @llvm-project

// NOLINTNEXTLINE(misc-include-cleaner): Required to define MathExtOps

// Generated definitions
#include "lib/Dialect/MathExt/IR/MathExtDialect.cpp.inc"

#define GET_OP_CLASSES
#include "lib/Dialect/MathExt/IR/MathExtOps.cpp.inc"

namespace mlir {
namespace heir {
namespace math_ext {

// https://github.com/llvm/llvm-project/pull/140171/files

// struct OneHotOpBufferization
//     : public OpInterface<OneHotOp,
//                              bufferization::BufferizableOpInterface> {
struct OneHotOpBufferization
    : public bufferization::BufferizableOpInterface::ExternalModel<
          OneHotOpBufferization, OneHotOp> {
  // using OpInterface<OneHotOp,
  //                       bufferization::BufferizableOpInterface>::OpInterface;

  LogicalResult bufferize(
      Operation *op, RewriterBase &rewriter,
      const bufferization::BufferizationOptions &options) const {
    auto oneHotOp = cast<OneHotOp>(op);
    Location loc = oneHotOp.getLoc();

    // Use dyn_cast for safer type casting
    auto resultTensorType =
        dyn_cast<RankedTensorType>(oneHotOp.getResult().getType());
    if (!resultTensorType) {
      return op->emitOpError("expected result to be a ranked tensor");
    }

    auto memRefType = MemRefType::get(resultTensorType.getShape(),
                                      resultTensorType.getElementType());

    Value allocatedMemRef = memref::AllocOp::create(rewriter, loc, memRefType);

    Value falseValue = arith::ConstantOp::create(
        rewriter, loc, rewriter.getIntegerAttr(rewriter.getI1Type(), 0));
    Value trueValue = arith::ConstantOp::create(
        rewriter, loc, rewriter.getIntegerAttr(rewriter.getI1Type(), 1));

    linalg::FillOp::create(rewriter, loc, falseValue, allocatedMemRef);

    Value index = arith::IndexCastOp::create(
        rewriter, loc, rewriter.getIndexType(), oneHotOp.getValue());
    memref::StoreOp::create(rewriter, loc, trueValue, allocatedMemRef,
                            ValueRange{index});

    bufferization::replaceOpWithBufferizedValues(rewriter, op, allocatedMemRef);

    return success();
  }
};

// Register the interface implementation with the dialect.
void MathExtDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "lib/Dialect/MathExt/IR/MathExtOps.cpp.inc"
      >();
}

}  // namespace math_ext
}  // namespace heir
}  // namespace mlir
