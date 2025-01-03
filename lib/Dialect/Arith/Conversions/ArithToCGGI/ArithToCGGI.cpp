#include "lib/Dialect/Arith/Conversions/ArithToCGGI/ArithToCGGI.h"

#include "lib/Dialect/CGGI/IR/CGGIDialect.h"
#include "lib/Dialect/CGGI/IR/CGGIOps.h"
#include "lib/Dialect/LWE/IR/LWEOps.h"
#include "lib/Dialect/LWE/IR/LWETypes.h"
#include "lib/Utils/ConversionUtils.h"
#include "llvm/include/llvm/Support/Debug.h"  // from @llvm-project
#include "mlir/include/mlir/Dialect/Affine/IR/AffineOps.h"  // from @llvm-project
#include "mlir/include/mlir/Dialect/Arith/IR/Arith.h"    // from @llvm-project
#include "mlir/include/mlir/Dialect/MemRef/IR/MemRef.h"  // from @llvm-project
#include "mlir/include/mlir/Transforms/DialectConversion.h"  // from @llvm-project

namespace mlir::heir::arith {

#define GEN_PASS_DEF_ARITHTOCGGI
#include "lib/Dialect/Arith/Conversions/ArithToCGGI/ArithToCGGI.h.inc"

static lwe::LWECiphertextType convertArithType(IntegerType type,
                                               MLIRContext *ctx) {
  return lwe::LWECiphertextType::get(ctx,
                                     lwe::UnspecifiedBitFieldEncodingAttr::get(
                                         ctx, type.getIntOrFloatBitWidth()),
                                     lwe::LWEParamsAttr());
  ;
}

static Type convertArithLikeType(ShapedType type, MLIRContext *ctx) {
  if (auto arithType = llvm::dyn_cast<IntegerType>(type.getElementType())) {
    return type.cloneWith(type.getShape(), convertArithType(arithType, ctx));
  }
  return type;
}

// Remove this class if no type conversions are necessary
class ArithToCGGITypeConverter : public TypeConverter {
 public:
  ArithToCGGITypeConverter(MLIRContext *ctx) {
    addConversion([](Type type) { return type; });

    // Convert Integer types to LWE ciphertext types
    addConversion([ctx](IntegerType type) -> Type {
      return convertArithType(type, ctx);
    });

    addConversion([ctx](ShapedType type) -> Type {
      return convertArithLikeType(type, ctx);
    });
    // addTargetMaterialization(buildConstantOp);
  }
};

class SecretTypeConverter : public TypeConverter {
 public:
  SecretTypeConverter(MLIRContext *ctx, int minBitWidth)
      : minBitWidth(minBitWidth) {
    addConversion([](Type type) { return type; });
  }

  int minBitWidth;
};

struct ConvertConstantOp : public OpConversionPattern<mlir::arith::ConstantOp> {
  ConvertConstantOp(mlir::MLIRContext *context)
      : OpConversionPattern<mlir::arith::ConstantOp>(context) {}

  using OpConversionPattern::OpConversionPattern;

  LogicalResult matchAndRewrite(
      mlir::arith::ConstantOp op, OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
    if (isa<IndexType>(op.getValue().getType())) {
      return failure();
    }
    ImplicitLocOpBuilder b(op.getLoc(), rewriter);

    auto intValue = cast<IntegerAttr>(op.getValue()).getValue().getSExtValue();
    auto inputValue = mlir::IntegerAttr::get(op.getType(), intValue);

    auto encoding = lwe::UnspecifiedBitFieldEncodingAttr::get(
        op->getContext(), op.getValue().getType().getIntOrFloatBitWidth());
    auto lweType = lwe::LWECiphertextType::get(op->getContext(), encoding,
                                               lwe::LWEParamsAttr());

    auto encrypt = b.create<cggi::CreateTrivialOp>(lweType, inputValue);

    rewriter.replaceOp(op, encrypt);
    return success();
  }
};

struct ConvertTruncIOp : public OpConversionPattern<mlir::arith::TruncIOp> {
  ConvertTruncIOp(mlir::MLIRContext *context)
      : OpConversionPattern<mlir::arith::TruncIOp>(context) {}

  using OpConversionPattern::OpConversionPattern;

  LogicalResult matchAndRewrite(
      mlir::arith::TruncIOp op, OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
    ImplicitLocOpBuilder b(op.getLoc(), rewriter);

    // Get both sizes, find difference, implement a ShR
    auto resultType = op.getResult().getType();
    auto inputType = op.getOperand().getType();

    auto shiftAmount =
        inputType.getIntOrFloatBitWidth() - resultType.getIntOrFloatBitWidth();

    auto inputValue = mlir::IntegerAttr::get(rewriter.getI8Type(), shiftAmount);

    auto cteOp = rewriter.create<mlir::arith::ConstantOp>(
        op.getLoc(), rewriter.getI8Type(), inputValue);

    auto shiftOp =
        b.create<cggi::ShiftRightOp>(resultType, op.getOperand(), cteOp);

    rewriter.replaceOp(op, shiftOp);
    return success();
  }
};

struct ConvertExtUIOp : public OpConversionPattern<mlir::arith::ExtUIOp> {
  ConvertExtUIOp(mlir::MLIRContext *context)
      : OpConversionPattern<mlir::arith::ExtUIOp>(context) {}

  using OpConversionPattern::OpConversionPattern;

  LogicalResult matchAndRewrite(
      mlir::arith::ExtUIOp op, OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
    ImplicitLocOpBuilder b(op.getLoc(), rewriter);

    auto

        // auto castOp = b.create<cggi::CastOp>(
        //     op.getLoc(), convertArithType(op.getType()), adaptor.getIn());
        // auto castOp = b.create<cggi::CastOp>(op.getResult().getType(),
        // op.getOperand());

        rewriter.replaceOp(op, castOp);
    return success();
  }
};

struct ArithToCGGI : public impl::ArithToCGGIBase<ArithToCGGI> {
  void runOnOperation() override {
    MLIRContext *context = &getContext();
    auto *module = getOperation();
    ArithToCGGITypeConverter typeConverter(context);

    RewritePatternSet patterns(context);
    ConversionTarget target(*context);
    target.addLegalDialect<cggi::CGGIDialect>();
    target.addIllegalDialect<mlir::arith::ArithDialect>();

    target.addDynamicallyLegalOp<mlir::arith::ConstantOp>(
        [](mlir::arith::ConstantOp op) {
          return isa<IndexType>(op.getValue().getType());
        });

    target.addDynamicallyLegalOp<
        memref::AllocOp, memref::DeallocOp, memref::StoreOp, memref::SubViewOp,
        memref::CopyOp, tensor::FromElementsOp, tensor::ExtractOp,
        affine::AffineStoreOp, affine::AffineLoadOp>([&](Operation *op) {
      return typeConverter.isLegal(op->getOperandTypes()) &&
             typeConverter.isLegal(op->getResultTypes());
    });

    patterns.add<
        ConvertConstantOp, ConvertTruncIOp,
        ConvertBinOp<mlir::arith::AddIOp, cggi::AddOp>,
        ConvertBinOp<mlir::arith::MulIOp, cggi::MulOp>,
        ConvertBinOp<mlir::arith::SubIOp, cggi::SubOp>,
        ConvertAny<memref::LoadOp>, ConvertAny<memref::AllocOp>,
        ConvertAny<memref::DeallocOp>, ConvertAny<memref::StoreOp>,
        ConvertAny<memref::SubViewOp>, ConvertAny<memref::CopyOp>,
        ConvertAny<tensor::FromElementsOp>, ConvertAny<tensor::ExtractOp>,
        ConvertAny<affine::AffineStoreOp>, ConvertAny<affine::AffineLoadOp> >(
        typeConverter, context);

    addStructuralConversionPatterns(typeConverter, patterns, target);

    if (failed(applyPartialConversion(module, target, std::move(patterns)))) {
      return signalPassFailure();
    }
  }
};

}  // namespace mlir::heir::arith
