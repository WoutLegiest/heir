#include "lib/Dialect/Arith/Conversions/ArithToCGGIQuart/ArithToCGGIQuart.h"

#include "lib/Dialect/CGGI/IR/CGGIDialect.h"
#include "lib/Dialect/CGGI/IR/CGGIOps.h"
#include "lib/Dialect/LWE/IR/LWEOps.h"
#include "lib/Dialect/LWE/IR/LWETypes.h"
#include "lib/Utils/ConversionUtils.h"
#include "llvm/include/llvm/Support/Debug.h"           // from @llvm-project
#include "llvm/include/llvm/Support/FormatVariadic.h"  // from @llvm-project
#include "mlir/include/mlir/Dialect/Affine/IR/AffineOps.h"  // from @llvm-project
#include "mlir/include/mlir/Dialect/Arith/IR/Arith.h"    // from @llvm-project
#include "mlir/include/mlir/Dialect/MemRef/IR/MemRef.h"  // from @llvm-project
#include "mlir/include/mlir/Dialect/Tensor/IR/Tensor.h"  // from @llvm-project
#include "mlir/include/mlir/Pass/PassManager.h"          // from @llvm-project
#include "mlir/include/mlir/Transforms/DialectConversion.h"  // from @llvm-project
#include "mlir/include/mlir/Transforms/Passes.h"  // from @llvm-project

namespace mlir::heir::arith {

#define GEN_PASS_DEF_ARITHTOCGGIQUART
#include "lib/Dialect/Arith/Conversions/ArithToCGGIQuart/ArithToCGGIQuart.h.inc"

static constexpr unsigned maxIntWidth = 16;

static lwe::LWECiphertextType convertArithToCGGIType(IntegerType type,
                                                     MLIRContext *ctx) {
  return lwe::LWECiphertextType::get(ctx,
                                     lwe::UnspecifiedBitFieldEncodingAttr::get(
                                         ctx, type.getIntOrFloatBitWidth()),
                                     lwe::LWEParamsAttr());
}

static std::optional<Type> convertArithToCGGIQuartType(IntegerType type,
                                                       MLIRContext *ctx) {
  auto lweType = lwe::LWECiphertextType::get(
      ctx, lwe::UnspecifiedBitFieldEncodingAttr::get(ctx, maxIntWidth),
      lwe::LWEParamsAttr());

  float width = type.getWidth();
  float realWidth = maxIntWidth >> 1;

  uint8_t nbChunks = ceil(width / realWidth);

  if (width > 64) return std::nullopt;

  return RankedTensorType::get({nbChunks}, lweType);
}

static std::optional<Type> convertArithLikeToCGGIQuartType(ShapedType type,
                                                           MLIRContext *ctx) {
  if (auto arithType = llvm::dyn_cast<IntegerType>(type.getElementType())) {
    float width = arithType.getWidth();
    float realWidth = maxIntWidth >> 1;

    uint8_t nbChunks = ceil(width / realWidth);

    if (width > 64) return std::nullopt;

    if (arithType.getIntOrFloatBitWidth() == maxIntWidth)
      return convertArithToCGGIQuartType(arithType, ctx);

    auto newShape = to_vector(type.getShape());
    newShape.push_back(nbChunks);

    if (isa<RankedTensorType>(type)) {
      return RankedTensorType::get(
          newShape, IntegerType::get(type.getContext(), maxIntWidth));
    }

    if (isa<MemRefType>(type)) {
      return MemRefType::get(newShape,
                             IntegerType::get(type.getContext(), maxIntWidth));
    }
  }
  return type;
}

class ArithToCGGIQuartTypeConverter : public TypeConverter {
 public:
  ArithToCGGIQuartTypeConverter(MLIRContext *ctx) {
    addConversion([](Type type) { return type; });

    // Convert Integer types to LWE ciphertext types
    addConversion([ctx](IntegerType type) -> std::optional<Type> {
      return convertArithToCGGIQuartType(type, ctx);
    });

    addConversion([ctx](ShapedType type) -> std::optional<Type> {
      return convertArithLikeToCGGIQuartType(type, ctx);
    });
  }
};

static Value createTrivialOpMaxWidth(ImplicitLocOpBuilder b, int value) {
  auto maxWideIntType = IntegerType::get(b.getContext(), maxIntWidth);
  auto intAttr = b.getIntegerAttr(maxWideIntType, value);

  auto encoding =
      lwe::UnspecifiedBitFieldEncodingAttr::get(b.getContext(), maxIntWidth);
  auto lweType = lwe::LWECiphertextType::get(b.getContext(), encoding,
                                             lwe::LWEParamsAttr());

  return b.create<cggi::CreateTrivialOp>(lweType, intAttr);
}

/// Extracts the `input` tensor slice with elements at the last dimension offset
/// by `lastOffset`. Returns a value of tensor type with the last dimension
/// reduced to x1 or fully scalarized, e.g.:
///   - tensor<2xi16>   --> i16
static Value extractLastDimSlice(ConversionPatternRewriter &rewriter,
                                 Location loc, Value input,
                                 int64_t lastOffset) {
  ArrayRef<int64_t> shape = cast<RankedTensorType>(input.getType()).getShape();
  assert(lastOffset < shape.back() && "Offset out of bounds");

  // Create index element
  auto intAttr = rewriter.getIntegerAttr(rewriter.getIndexType(), lastOffset);
  auto constantOp = rewriter.create<mlir::arith::ConstantOp>(loc, intAttr);
  SmallVector<Value, 1> indices;
  indices.push_back(constantOp.getResult());

  // Scalarize the result in case of 1D tensors.
  if (shape.size() == 1) {
    return rewriter.create<tensor::ExtractOp>(loc, input, indices);
  }

  SmallVector<OpFoldResult> offsets(shape.size(), rewriter.getIndexAttr(0));
  offsets.back() = rewriter.getIndexAttr(lastOffset);
  SmallVector<OpFoldResult> sizes(shape.size());
  sizes.back() = rewriter.getIndexAttr(1);
  SmallVector<OpFoldResult> strides(shape.size(), rewriter.getIndexAttr(1));

  return rewriter.create<tensor::ExtractSliceOp>(loc, input, offsets, sizes,
                                                 strides);
}

/// Extracts four tensor slices from the `input` whose type is `tensor<...x4T>`,
/// with the first element at offset 0, second element at offset 1 and so on.
static SmallVector<Value> extractLastDimHalves(
    ConversionPatternRewriter &rewriter, Location loc, Value input) {
  auto tenShape = cast<ShapedType>(input.getType()).getShape();
  auto nbChunks = tenShape.back();
  SmallVector<Value> newTrivialOps;

  for (int i = 0; i < nbChunks; ++i) {
    newTrivialOps.push_back(extractLastDimSlice(rewriter, loc, input, i));
  }

  return newTrivialOps;
};

static Value createScalarOrSplatConstant(OpBuilder &builder, Location loc,
                                         Type type, int64_t value) {
  unsigned elementBitWidth = 0;
  if (auto lweTy = dyn_cast<lwe::LWECiphertextType>(type))
    elementBitWidth =
        cast<lwe::UnspecifiedBitFieldEncodingAttr>(lweTy.getEncoding())
            .getCleartextBitwidth();
  else
    elementBitWidth = maxIntWidth;

  auto apValue = APInt(elementBitWidth, value);

  auto maxWideIntType =
      IntegerType::get(builder.getContext(), maxIntWidth >> 1);
  auto intAttr = builder.getIntegerAttr(maxWideIntType, value);

  return builder.create<cggi::CreateTrivialOp>(loc, type, intAttr);
}

static Value insertLastDimSlice(ConversionPatternRewriter &rewriter,
                                Location loc, Value source, Value dest,
                                int64_t lastOffset) {
  ArrayRef<int64_t> shape = cast<RankedTensorType>(dest.getType()).getShape();
  assert(lastOffset < shape.back() && "Offset out of bounds");

  // // Handle single element source.
  if (isa<lwe::LWECiphertextType>(source.getType())) {
    auto intAttr = rewriter.getIntegerAttr(rewriter.getIndexType(), lastOffset);
    auto constantOp = rewriter.create<mlir::arith::ConstantOp>(loc, intAttr);
    SmallVector<Value, 1> indices;
    indices.push_back(constantOp.getResult());

    return rewriter.create<tensor::InsertOp>(loc, source, dest, indices);
  }

  SmallVector<OpFoldResult> offsets(shape.size(), rewriter.getIndexAttr(0));
  offsets.back() = rewriter.getIndexAttr(lastOffset);
  SmallVector<OpFoldResult> sizes(shape.size());
  sizes.back() = rewriter.getIndexAttr(1);
  SmallVector<OpFoldResult> strides(shape.size(), rewriter.getIndexAttr(1));

  return rewriter.create<tensor::InsertSliceOp>(loc, source, dest, offsets,
                                                sizes, strides);
}

/// Constructs a new tensor of type `resultType` by creating a series of
/// insertions of `resultComponents`, each at the next offset of the last tensor
/// dimension.
/// When all `resultComponents` are scalars, the result type is `tensor<NxT>`;
/// when `resultComponents` are `tensor<...x1xT>`s, the result type is
/// `tensor<...xNxT>`, where `N` is the number of `resultComponents`.
static Value constructResultTensor(ConversionPatternRewriter &rewriter,
                                   Location loc, RankedTensorType resultType,
                                   ValueRange resultComponents) {
  Value resultVec = createScalarOrSplatConstant(rewriter, loc, resultType, 0);
  for (auto [i, component] : llvm::enumerate(resultComponents))
    resultVec = insertLastDimSlice(rewriter, loc, component, resultVec, i);

  return resultVec;
}

struct ConvertQuartConstantOp
    : public OpConversionPattern<mlir::arith::ConstantOp> {
  ConvertQuartConstantOp(mlir::MLIRContext *context)
      : OpConversionPattern<mlir::arith::ConstantOp>(context) {}

  using OpConversionPattern::OpConversionPattern;

  LogicalResult matchAndRewrite(
      mlir::arith::ConstantOp op, OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
    if (isa<IndexType>(op.getValue().getType())) {
      return failure();
    }
    ImplicitLocOpBuilder b(op.getLoc(), rewriter);

    Type oldType = op.getType();
    auto newType = getTypeConverter()->convertType<RankedTensorType>(oldType);
    auto acutalBitWidth = maxIntWidth >> 1;

    if (!newType)
      return rewriter.notifyMatchFailure(
          op, llvm::formatv("unsupported type: {0}", op.getType()));

    Attribute oldValue = op.getValueAttr();
    auto tenShape = newType.getShape();
    auto nbChunks = tenShape.back();
    SmallVector<Value, 1> newTrivialOps;

    if (auto intAttr = dyn_cast<IntegerAttr>(oldValue)) {
      for (uint8_t i = 0; i < nbChunks; i++) {
        APInt intChunck =
            intAttr.getValue().extractBits(acutalBitWidth, i * acutalBitWidth);

        auto encrypt = createTrivialOpMaxWidth(b, intChunck.getSExtValue());
        newTrivialOps.push_back(encrypt);
      }

      Value resultVec =
          constructResultTensor(rewriter, op.getLoc(), newType, newTrivialOps);
      rewriter.replaceOp(op, resultVec);

      return success();
    }
  }
};

struct ConvertQuartTruncIOp
    : public OpConversionPattern<mlir::arith::TruncIOp> {
  ConvertQuartTruncIOp(mlir::MLIRContext *context)
      : OpConversionPattern<mlir::arith::TruncIOp>(context) {}

  using OpConversionPattern::OpConversionPattern;

  LogicalResult matchAndRewrite(
      mlir::arith::TruncIOp op, OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
    ImplicitLocOpBuilder b(op.getLoc(), rewriter);

    auto newResultTy = getTypeConverter()->convertType<RankedTensorType>(
        op.getResult().getType());
    auto newInTy =
        getTypeConverter()->convertType<RankedTensorType>(op.getIn().getType());

    SmallVector<OpFoldResult> offsets(newResultTy.getShape().size(),
                                      rewriter.getIndexAttr(0));
    offsets.back() = rewriter.getIndexAttr(newInTy.getShape().back() -
                                           newResultTy.getShape().back());
    SmallVector<OpFoldResult> sizes(newResultTy.getShape().size());
    sizes.back() = rewriter.getIndexAttr(1);
    SmallVector<OpFoldResult> strides(newResultTy.getShape().size(),
                                      rewriter.getIndexAttr(1));

    auto resOp = rewriter.create<tensor::ExtractSliceOp>(
        op->getLoc(), adaptor.getIn(), offsets, sizes, strides);
    rewriter.replaceOp(op, resOp);

    return success();
  }
};

template <typename ArithExtOp>
struct ConvertQuartExt final : OpConversionPattern<ArithExtOp> {
  using OpConversionPattern<ArithExtOp>::OpConversionPattern;

  // Since each type inside the program is a tensor with 4 elements, we can
  // simply return the input tensor as the result. The generated code will later
  // be removed by the CSE pass.

  LogicalResult matchAndRewrite(
      ArithExtOp op, typename ArithExtOp::Adaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
    ImplicitLocOpBuilder b(op->getLoc(), rewriter);

    auto newResultTy = cast<ShapedType>(
        convertArithToCGGIQuartType(cast<IntegerType>(op.getResult().getType()),
                                    op.getContext())
            .value());
    auto newInTy = cast<ShapedType>(
        convertArithToCGGIQuartType(cast<IntegerType>(op.getIn().getType()),
                                    op.getContext())
            .value());

    auto resultChunks = newResultTy.getShape().back();
    auto inChunks = newInTy.getShape().back();

    // Through definition of ExtOp, paddingFactor is always positive
    auto paddingFactor = resultChunks - inChunks;

    SmallVector<OpFoldResult, 1> low, high;
    low.push_back(rewriter.getIndexAttr(0));
    high.push_back(rewriter.getIndexAttr(paddingFactor));

    auto padValue = createTrivialOpMaxWidth(b, 0);

    auto resultVec = b.create<tensor::PadOp>(newResultTy, adaptor.getIn(), low,
                                             high, padValue,
                                             /*nofold=*/true);

    rewriter.replaceOp(op, resultVec);
    return success();
  }
};

struct ConvertQuartAddI final : OpConversionPattern<mlir::arith::AddIOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult matchAndRewrite(
      mlir::arith::AddIOp op, OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
    Location loc = op->getLoc();
    ImplicitLocOpBuilder b(loc, rewriter);

    auto newTy =
        getTypeConverter()->convertType<RankedTensorType>(op.getType());
    if (!newTy)
      return rewriter.notifyMatchFailure(
          loc, llvm::formatv("unsupported type: {0}", op.getType()));

    SmallVector<Value> splitLhs =
        extractLastDimHalves(rewriter, loc, adaptor.getLhs());
    SmallVector<Value> splitRhs =
        extractLastDimHalves(rewriter, loc, adaptor.getRhs());

    assert(splitLhs.size() == splitRhs.size() && "Mismatched tensor sizes");

    // Actual type of the underlying elements; we use half the width.
    // Create Constant
    auto shiftAttr =
        IntegerAttr::get(rewriter.getIndexType(), maxIntWidth >> 1);

    auto elemType = convertArithToCGGIType(
        IntegerType::get(op->getContext(), maxIntWidth), op->getContext());
    auto realTy = convertArithToCGGIType(
        IntegerType::get(op->getContext(), maxIntWidth >> 1), op->getContext());

    // auto constantOp = b.create<mlir::arith::ConstantOp>(intAttr);

    SmallVector<Value> carries;
    SmallVector<Value> outputs;

    for (int i = 0; i < splitLhs.size(); ++i) {
      auto lowSum = b.create<cggi::AddOp>(splitLhs[i], splitRhs[i]);
      auto outputLsb = b.create<cggi::CastOp>(op.getLoc(), realTy, lowSum);
      auto outputLsbHigh =
          b.create<cggi::CastOp>(op.getLoc(), elemType, outputLsb);

      // Now all the outputs are 16b elements, wants presentation of 4x8b
      if (i != splitLhs.size() - 1) {
        auto carry =
            b.create<cggi::ScalarShiftRightOp>(elemType, lowSum, shiftAttr);
        carries.push_back(carry);
      }

      if (i == 0) {
        outputs.push_back(outputLsbHigh);
      } else {
        auto high = b.create<cggi::AddOp>(outputLsbHigh, carries[i - 1]);
        outputs.push_back(high);
      }
    }

    Value resultVec = constructResultTensor(rewriter, loc, newTy, outputs);
    rewriter.replaceOp(op, resultVec);
    return success();
  }
};

// Implemented using the Karatsuba algorithm
// https://en.wikipedia.org/wiki/Karatsuba_algorithm#Algorithm
struct ConvertQuartMulI final : OpConversionPattern<mlir::arith::MulIOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult matchAndRewrite(
      mlir::arith::MulIOp op, OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
    Location loc = op->getLoc();
    ImplicitLocOpBuilder b(loc, rewriter);

    auto newTy =
        getTypeConverter()->convertType<RankedTensorType>(op.getType());
    if (!newTy)
      return rewriter.notifyMatchFailure(
          loc, llvm::formatv("unsupported type: {0}", op.getType()));
    if (newTy.getShape().back() != 4)
      return rewriter.notifyMatchFailure(
          loc, llvm::formatv("Mul only support 4 split elements. Shape: {0}",
                             newTy));

    auto elemTy = convertArithToCGGIType(
        IntegerType::get(op->getContext(), maxIntWidth), op->getContext());
    auto realTy = convertArithToCGGIType(
        IntegerType::get(op->getContext(), maxIntWidth >> 1), op->getContext());

    // Create Constant
    auto shiftAttr =
        rewriter.getIntegerAttr(b.getIndexType(), maxIntWidth >> 1);

    SmallVector<Value> splitLhs =
        extractLastDimHalves(rewriter, loc, adaptor.getLhs());
    SmallVector<Value> splitRhs =
        extractLastDimHalves(rewriter, loc, adaptor.getRhs());

    // TODO: Implement the real Karatsuba algorithm for 4x4 multiplication.
    // First part of Karatsuba algorithm
    auto z00 = b.create<cggi::MulOp>(splitLhs[0], splitRhs[0]);
    auto z02 = b.create<cggi::MulOp>(splitLhs[1], splitRhs[1]);
    auto z01_p1 = b.create<cggi::AddOp>(splitLhs[0], splitLhs[1]);
    auto z01_p2 = b.create<cggi::AddOp>(splitRhs[0], splitRhs[1]);
    auto z01_m = b.create<cggi::MulOp>(z01_p1, z01_p2);
    auto z01_s = b.create<cggi::SubOp>(z01_m, z00);
    auto z01 = b.create<cggi::SubOp>(z01_s, z02);

    // Second part I of Karatsuba algorithm
    auto z1a0 = b.create<cggi::MulOp>(splitLhs[0], splitRhs[2]);
    auto z1a2 = b.create<cggi::MulOp>(splitLhs[1], splitRhs[3]);
    auto z1a1_p1 = b.create<cggi::AddOp>(splitLhs[0], splitLhs[1]);
    auto z1a1_p2 = b.create<cggi::AddOp>(splitRhs[2], splitRhs[3]);
    auto z1a1_m = b.create<cggi::MulOp>(z1a1_p1, z1a1_p2);
    auto z1a1_s = b.create<cggi::SubOp>(z1a1_m, z1a0);
    auto z1a1 = b.create<cggi::SubOp>(z1a1_s, z1a2);

    // Second part II of Karatsuba algorithm
    auto z1b0 = b.create<cggi::MulOp>(splitLhs[2], splitRhs[0]);
    auto z1b2 = b.create<cggi::MulOp>(splitLhs[3], splitRhs[1]);
    auto z1b1_p1 = b.create<cggi::AddOp>(splitLhs[2], splitLhs[3]);
    auto z1b1_p2 = b.create<cggi::AddOp>(splitRhs[0], splitRhs[1]);
    auto z1b1_m = b.create<cggi::MulOp>(z1b1_p1, z1b1_p2);
    auto z1b1_s = b.create<cggi::SubOp>(z1b1_m, z1b0);
    auto z1b1 = b.create<cggi::SubOp>(z1b1_s, z1b2);

    auto out2Kara = b.create<cggi::AddOp>(z1a0, z1b0);
    auto out2Carry = b.create<cggi::AddOp>(out2Kara, z02);
    auto out3Carry = b.create<cggi::AddOp>(z1a1, z1b1);

    // Output are now all 16b elements, wants presentation of 4x8b
    auto output0Lsb = b.create<cggi::CastOp>(realTy, z00);
    auto output0LsbHigh = b.create<cggi::CastOp>(elemTy, output0Lsb);
    auto output0Msb =
        b.create<cggi::ScalarShiftRightOp>(elemTy, z00, shiftAttr);

    auto output1Lsb = b.create<cggi::CastOp>(realTy, z01);
    auto output1LsbHigh = b.create<cggi::CastOp>(elemTy, output1Lsb);
    auto output1Msb =
        b.create<cggi::ScalarShiftRightOp>(elemTy, z01, shiftAttr);

    auto output2Lsb = b.create<cggi::CastOp>(realTy, out2Carry);
    auto output2LsbHigh = b.create<cggi::CastOp>(elemTy, output2Lsb);
    auto output2Msb =
        b.create<cggi::ScalarShiftRightOp>(elemTy, out2Carry, shiftAttr);

    auto output3Lsb = b.create<cggi::CastOp>(realTy, out3Carry);
    auto output3LsbHigh = b.create<cggi::CastOp>(elemTy, output3Lsb);

    auto output1 = b.create<cggi::AddOp>(output1LsbHigh, output0Msb);
    auto output2 = b.create<cggi::AddOp>(output2LsbHigh, output1Msb);
    auto output3 = b.create<cggi::AddOp>(output3LsbHigh, output2Msb);

    Value resultVec = constructResultTensor(
        rewriter, loc, newTy, {output0LsbHigh, output1, output2, output3});
    rewriter.replaceOp(op, resultVec);
    return success();
  }
};

struct ArithToCGGIQuart : public impl::ArithToCGGIQuartBase<ArithToCGGIQuart> {
  void runOnOperation() override {
    MLIRContext *context = &getContext();
    auto *module = getOperation();
    ArithToCGGIQuartTypeConverter typeConverter(context);

    RewritePatternSet patterns(context);
    ConversionTarget target(*context);
    target.addLegalDialect<cggi::CGGIDialect>();
    target.addLegalDialect<tensor::TensorDialect>();
    target.addLegalDialect<memref::MemRefDialect>();

    auto opLegalCallback = [&typeConverter](Operation *op) {
      return typeConverter.isLegal(op);
    };

    target.addDynamicallyLegalOp<func::CallOp, func::ReturnOp>(opLegalCallback);
    target.addDynamicallyLegalDialect<mlir::arith::ArithDialect,
                                      tensor::TensorDialect>(opLegalCallback);

    target.addDynamicallyLegalOp<
        memref::AllocOp, memref::DeallocOp, memref::StoreOp, memref::LoadOp,
        memref::SubViewOp, memref::CopyOp, affine::AffineLoadOp,
        affine::AffineStoreOp, tensor::FromElementsOp, tensor::ExtractOp>(
        [&](Operation *op) {
          return typeConverter.isLegal(op->getOperandTypes()) &&
                 typeConverter.isLegal(op->getResultTypes());
        });

    target.addDynamicallyLegalOp<mlir::arith::ConstantOp>(
        [](mlir::arith::ConstantOp op) {
          return isa<IndexType>(op.getValue().getType());
        });

    patterns
        .add<ConvertQuartConstantOp, ConvertQuartExt<mlir::arith::ExtUIOp>,
             ConvertQuartExt<mlir::arith::ExtSIOp>, ConvertQuartAddI,
             ConvertQuartMulI, ConvertQuartTruncIOp, ConvertAny<memref::LoadOp>,
             ConvertAny<memref::AllocOp>, ConvertAny<memref::DeallocOp>,
             ConvertAny<memref::StoreOp>, ConvertAny<memref::SubViewOp>,
             ConvertAny<memref::CopyOp>, ConvertAny<tensor::FromElementsOp>,
             ConvertAny<tensor::ExtractOp>, ConvertAny<affine::AffineStoreOp>,
             ConvertAny<affine::AffineLoadOp> >(typeConverter, context);

    addStructuralConversionPatterns(typeConverter, patterns, target);

    if (failed(applyPartialConversion(module, target, std::move(patterns)))) {
      return signalPassFailure();
    }

    // Remove the uncessary tensor ops between each converted arith operation.
    OpPassManager pipeline("builtin.module");
    pipeline.addPass(createCSEPass());
    (void)runPipeline(pipeline, getOperation());
  }
};

}  // namespace mlir::heir::arith
