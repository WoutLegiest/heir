#include "lib/Dialect/MathExt/Transforms/BufferizableOpInterfaceImpl.h"

#include "lib/Dialect/MathExt/IR/MathExtOps.h"
#include "mlir/include/mlir/Dialect/Arith/IR/Arith.h"  // from @llvm-project
#include "mlir/include/mlir/Dialect/Bufferization/IR/BufferizableOpInterface.h"  // from @llvm-project
#include "mlir/include/mlir/Dialect/Bufferization/Transforms/Bufferize.h"  // from @llvm-project
#include "mlir/include/mlir/Dialect/Linalg/IR/Linalg.h"  // Needed for linalg::FillOp
#include "mlir/include/mlir/Dialect/MemRef/IR/MemRef.h"  // from @llvm-project
#include "mlir/include/mlir/IR/Builders.h"               // from @llvm-project
#include "mlir/include/mlir/IR/BuiltinTypes.h"           // from @llvm-project
#include "mlir/include/mlir/IR/DialectImplementation.h"  // from @llvm-project
#include "mlir/include/mlir/IR/OpDefinition.h"           // from @llvm-project

using namespace mlir;
using namespace mlir::bufferization;

namespace mlir {
namespace heir {
namespace math_ext {

struct OneHotOpInterface
    : public bufferization::BufferizableOpInterface::ExternalModel<
          OneHotOpInterface, math_ext::OneHotOp> {
  // explicit OneHotOpBufferization(MLIRContext * /*ctx*/) {}

  bool bufferizesToAllocation(Operation *op, Value value) const {
    // This op always allocates a new buffer for its result.
    return true;
  }

  // OneHotOp has no tensor operands, so it never reads from an input buffer.
  bool bufferizesToMemoryRead(Operation *op, OpOperand &opOperand,
                              const bufferization::AnalysisState &state) const {
    return false;
  }

  // OneHotOp has no tensor operands, so it never writes to an input buffer.
  bool bufferizesToMemoryWrite(
      Operation *op, OpOperand &opOperand,
      const bufferization::AnalysisState &state) const {
    return false;
  }

  // This is not an in-place op, so no result can alias an operand.
  bufferization::AliasingValueList getAliasingOpResult(
      Operation *op, OpOperand &opOperand,
      const bufferization::AnalysisState &state) const {
    return {};
  }

  // Since there is no operand-result alias, the relationship is Unknown.
  bufferization::BufferRelation bufferRelation(
      Operation *op, OpResult opResult,
      const bufferization::AnalysisState &state) const {
    return bufferization::BufferRelation::Unknown;
  }

  LogicalResult bufferize(Operation *op, RewriterBase &rewriter,
                          const bufferization::BufferizationOptions &options,
                          bufferization::BufferizationState &state) const {
    llvm::dbgs() << " Executing this thingy" << "\n";

    auto oneHotOp = cast<math_ext::OneHotOp>(op);
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

void registerBufferizableOpInterfaceExternalModels(DialectRegistry &registry) {
  registry.addExtension(
      +[](MLIRContext *ctx, math_ext::MathExtDialect *dialect) {
        OneHotOp::attachInterface<OneHotOpInterface>(*ctx);
      });
}

}  // namespace math_ext
}  // namespace heir
}  // namespace mlir
