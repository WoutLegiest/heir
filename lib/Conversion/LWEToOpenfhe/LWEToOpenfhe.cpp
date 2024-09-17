#include "lib/Conversion/LWEToOpenfhe/LWEToOpenfhe.h"

#include <cassert>
#include <cstddef>
#include <cstdint>
#include <utility>

#include "lib/Conversion/Utils.h"
#include "lib/Dialect/LWE/IR/LWEAttributes.h"
#include "lib/Dialect/LWE/IR/LWEOps.h"
#include "lib/Dialect/LWE/IR/LWETypes.h"
#include "lib/Dialect/Openfhe/IR/OpenfheDialect.h"
#include "lib/Dialect/Openfhe/IR/OpenfheOps.h"
#include "lib/Dialect/Openfhe/IR/OpenfheTypes.h"
#include "llvm/include/llvm/ADT/SmallVector.h"          // from @llvm-project
#include "llvm/include/llvm/ADT/TypeSwitch.h"           // from @llvm-project
#include "llvm/include/llvm/Support/Casting.h"          // from @llvm-project
#include "mlir/include/mlir/Dialect/Arith/IR/Arith.h"   // from @llvm-project
#include "mlir/include/mlir/Dialect/Func/IR/FuncOps.h"  // from @llvm-project
#include "mlir/include/mlir/IR/BuiltinAttributes.h"     // from @llvm-project
#include "mlir/include/mlir/IR/BuiltinTypes.h"          // from @llvm-project
#include "mlir/include/mlir/IR/ImplicitLocOpBuilder.h"  // from @llvm-project
#include "mlir/include/mlir/IR/PatternMatch.h"          // from @llvm-project
#include "mlir/include/mlir/IR/Visitors.h"              // from @llvm-project
#include "mlir/include/mlir/Support/LLVM.h"             // from @llvm-project
#include "mlir/include/mlir/Support/LogicalResult.h"    // from @llvm-project
#include "mlir/include/mlir/Transforms/DialectConversion.h"  // from @llvm-project

namespace mlir::heir::lwe {

FailureOr<Value> getContextualCryptoContext(Operation *op) {
  auto result = getContextualArgFromFunc<openfhe::CryptoContextType>(op);
  if (failed(result)) {
    return op->emitOpError()
           << "Found BGV op in a function without a public "
              "key argument. Did the AddCryptoContextArg pattern fail to run?";
  }
  return result.value();
}

LogicalResult ConvertEncryptOp::matchAndRewrite(
    lwe::RLWEEncryptOp op, OpAdaptor adaptor,
    ConversionPatternRewriter &rewriter) const {
  FailureOr<Value> result = getContextualCryptoContext(op.getOperation());
  if (failed(result)) return result;

  auto keyType = dyn_cast<lwe::RLWEPublicKeyType>(op.getKey().getType());
  if (!keyType)
    return op.emitError()
           << "OpenFHE only supports public key encryption for LWE.";

  Value cryptoContext = result.value();
  rewriter.replaceOp(op,
                     rewriter.create<openfhe::EncryptOp>(
                         op.getLoc(), op.getOutput().getType(), cryptoContext,
                         adaptor.getInput(), adaptor.getKey()));
  return success();
}

LogicalResult ConvertDecryptOp::matchAndRewrite(
    lwe::RLWEDecryptOp op, OpAdaptor adaptor,
    ConversionPatternRewriter &rewriter) const {
  FailureOr<Value> result = getContextualCryptoContext(op.getOperation());
  if (failed(result)) return result;

  Value cryptoContext = result.value();
  rewriter.replaceOp(op,
                     rewriter.create<openfhe::DecryptOp>(
                         op.getLoc(), op.getOutput().getType(), cryptoContext,
                         adaptor.getInput(), adaptor.getSecretKey()));
  return success();
}

// OpenFHE has a convention that all inputs to MakePackedPlaintext are
// std::vector<int64_t>, so we need to cast the input to that type.
LogicalResult ConvertEncodeOp::matchAndRewrite(
    lwe::RLWEEncodeOp op, OpAdaptor adaptor,
    ConversionPatternRewriter &rewriter) const {
  FailureOr<Value> result = getContextualCryptoContext(op.getOperation());
  if (failed(result)) return result;
  Value cryptoContext = result.value();

  auto tensorTy =
      mlir::dyn_cast<RankedTensorType>(adaptor.getInput().getType());
  // TODO(#646): support scalar inputs to encode.
  if (!tensorTy) return op.emitOpError() << "Unsupported input type";

  auto elementTy = tensorTy.getElementType();
  auto intTy = mlir::dyn_cast<IntegerType>(elementTy);
  if (!intTy)
    return op.emitOpError() << "Input element type must be an integer type";

  if (intTy.getWidth() > 64)
    return op.emitError() << "No supported packing technique for integers "
                             "bigger than 64 bits.";

  auto castedInput = adaptor.getInput();
  if (intTy.getWidth() < 64) {
    auto int64Ty = rewriter.getIntegerType(64);
    auto newTensorTy = RankedTensorType::get(tensorTy.getShape(), int64Ty);
    castedInput = rewriter.create<arith::ExtSIOp>(op.getLoc(), newTensorTy,
                                                  adaptor.getInput());
  }

  lwe::RLWEPlaintextType plaintextType = lwe::RLWEPlaintextType::get(
      op.getContext(), op.getEncoding(), op.getRing(), tensorTy);
  rewriter.replaceOpWithNewOp<openfhe::MakePackedPlaintextOp>(
      op, plaintextType, cryptoContext, castedInput);

  return success();
}

}  // namespace mlir::heir::lwe