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
#include "mlir/include/mlir/Transforms/DialectConversion.h"  // from @llvm-project

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

  // if (width < maxIntWidth) return lweType;

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
  llvm::dbgs() << "type: " << type << "\n";
  if (auto lweTy = dyn_cast<lwe::LWECiphertextType>(type))
    elementBitWidth =
        cast<lwe::UnspecifiedBitFieldEncodingAttr>(lweTy.getEncoding())
            .getCleartextBitwidth();
  else
    elementBitWidth = maxIntWidth;

  auto apValue = APInt(elementBitWidth, value);

  llvm::dbgs() << "apValue: " << apValue << "\n";

  // TypedAttr attr;
  // if (isa<IntegerType>(type)) {
  //   attr = builder.getIntegerAttr(type, apValue);
  // } else {
  //   auto vecTy = cast<ShapedType>(type);
  //   attr = SplatElementsAttr::get(vecTy, apValue);
  // }

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

  // // Handle scalar source.
  // if (isa<IntegerType>(source.getType())) {
  auto intAttr = rewriter.getIntegerAttr(rewriter.getIndexType(), lastOffset);
  auto constantOp = rewriter.create<mlir::arith::ConstantOp>(loc, intAttr);
  SmallVector<Value, 1> indices;
  indices.push_back(constantOp.getResult());

  return rewriter.create<tensor::InsertOp>(loc, source, dest, indices);
  // }

  // SmallVector<OpFoldResult> offsets(shape.size(), rewriter.getIndexAttr(0));
  // offsets.back() = rewriter.getIndexAttr(lastOffset);
  // SmallVector<OpFoldResult> sizes(shape.size());
  // sizes.back() = rewriter.getIndexAttr(1);
  // SmallVector<OpFoldResult> strides(shape.size(), rewriter.getIndexAttr(1));

  // return rewriter.create<tensor::InsertSliceOp>(loc, source, dest, offsets,
  //                                               sizes, strides);
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
  llvm::dbgs() << "resultVec: " << resultVec << "\n";
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

    llvm::dbgs() << "#########################\n";
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

    llvm::dbgs() << "#########################\n";
    llvm::dbgs() << "nbChunks: " << nbChunks << "\n";
    llvm::dbgs() << "newBitWidth: " << maxIntWidth << "\n";
    llvm::dbgs() << "acutalBitWidth: " << acutalBitWidth << "\n";
    llvm::dbgs() << "oldType: " << oldType << "\n";
    llvm::dbgs() << "newType: " << newType << "\n";

    auto encoding = lwe::UnspecifiedBitFieldEncodingAttr::get(op->getContext(),
                                                              maxIntWidth);
    auto lweType = lwe::LWECiphertextType::get(op->getContext(), encoding,
                                               lwe::LWEParamsAttr());
    auto maxWideIntType = IntegerType::get(op->getContext(), maxIntWidth);

    if (auto intAttr = dyn_cast<IntegerAttr>(oldValue)) {
      for (uint8_t i = 0; i < nbChunks; i++) {
        APInt intChunck =
            intAttr.getValue().extractBits(acutalBitWidth, i * acutalBitWidth);
        auto intAttr =
            IntegerAttr::get(maxWideIntType, intChunck.getSExtValue());

        llvm::dbgs() << "intChunck{" << i << "}: " << intChunck << "\n";

        auto encrypt = b.create<cggi::CreateTrivialOp>(lweType, intAttr);
        newTrivialOps.push_back(encrypt);
      }

      Value resultVec =
          constructResultTensor(rewriter, op.getLoc(), newType, newTrivialOps);
      rewriter.replaceOp(op, resultVec);

      return success();
    }

    // if (auto elemsAttr = dyn_cast<DenseElementsAttr>(oldValue)) {
    //   int64_t numElems = elemsAttr.getNumElements();
    //   SmallVector<APInt> values;
    //   values.reserve(numElems * nbChunks);
    //   for (const APInt &origVal : elemsAttr.getValues<APInt>()) {
    //     for (uint8_t i = 0; i < nbChunks; i++) {
    //       APInt intChunck =
    //           origVal.extractBits(acutalBitWidth, i * acutalBitWidth);
    //       values.push_back(intChunck);
    //     }
    //   }

    //   auto attr = DenseElementsAttr::get(newType, values);
    //   rewriter.replaceOpWithNewOp<mlir::arith::ConstantOp>(op, attr);
    //   return success();
    // }

    // return rewriter.notifyMatchFailure(op.getLoc(),
    //                                    "unhandled constant attribute");

    // auto intValue =
    // cast<IntegerAttr>(op.getValue()).getValue().getSExtValue(); auto
    // inputValue = mlir::IntegerAttr::get(op.getType(), intValue);

    // auto encoding = lwe::UnspecifiedBitFieldEncodingAttr::get(
    //     op->getContext(), op.getValue().getType().getIntOrFloatBitWidth());
    // auto lweType = lwe::LWECiphertextType::get(op->getContext(), encoding,
    //                                            lwe::LWEParamsAttr());

    // auto encrypt = b.create<cggi::CreateTrivialOp>(lweType, inputValue);

    // rewriter.replaceOp(op, encrypt);
    // return success();
  }
};

// struct ConvertTruncIOp : public OpConversionPattern<mlir::arith::TruncIOp> {
//   ConvertTruncIOp(mlir::MLIRContext *context)
//       : OpConversionPattern<mlir::arith::TruncIOp>(context) {}

//   using OpConversionPattern::OpConversionPattern;

//   LogicalResult matchAndRewrite(
//       mlir::arith::TruncIOp op, OpAdaptor adaptor,
//       ConversionPatternRewriter &rewriter) const override {
//     ImplicitLocOpBuilder b(op.getLoc(), rewriter);

//     auto outType = convertArithToCGGIQuartType(
//         cast<IntegerType>(op.getResult().getType()), op->getContext());
//     auto castOp = b.create<cggi::CastOp>(op.getLoc(), outType,
//     adaptor.getIn());

//     rewriter.replaceOp(op, castOp);
//     return success();
//   }
// };

struct ConvertQuartExtUI final : OpConversionPattern<mlir::arith::ExtUIOp> {
  using OpConversionPattern::OpConversionPattern;

  // Since each type inside the program is a tensor with 4 elements, we can
  // simply return the input tensor as the result. The generated code will later
  // be removed by the CSE pass.

  LogicalResult matchAndRewrite(
      mlir::arith::ExtUIOp op, OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
    ImplicitLocOpBuilder b(op->getLoc(), rewriter);

    auto newResultTy = getTypeConverter()->convertType<RankedTensorType>(
        op.getResult().getType());
    auto newInTy =
        getTypeConverter()->convertType<RankedTensorType>(op.getIn().getType());

    auto resultChunks = newResultTy.getShape().back();
    auto inChunks = newInTy.getShape().back();

    if (resultChunks > inChunks) {
      auto paddingFactor = ceil(resultChunks / inChunks);

      auto intAttrLow = rewriter.getIntegerAttr(rewriter.getIndexType(), 0);
      auto constantLowOp =
          rewriter.create<mlir::arith::ConstantOp>(op->getLoc(), intAttrLow);
      SmallVector<Value, 1> lowPadding;
      lowPadding.push_back(constantLowOp.getResult());

      auto intAttrHigh =
          rewriter.getIntegerAttr(rewriter.getIndexType(), paddingFactor);
      auto constantHighOp =
          rewriter.create<mlir::arith::ConstantOp>(op->getLoc(), intAttrHigh);
      SmallVector<Value, 1> highPadding;
      highPadding.push_back(constantHighOp.getResult());

      auto resultVec = b.create<tensor::PadOp>(newResultTy, adaptor.getIn(),
                                               lowPadding, highPadding);

      llvm::dbgs() << "ExtUI \n";
      llvm::dbgs() << "resultVec: " << resultVec << "\n";

      rewriter.replaceOp(op, resultVec);
      return success();
    }
  }
};

// struct ConvertExtSIOp : public OpConversionPattern<mlir::arith::ExtSIOp> {
//   ConvertExtSIOp(mlir::MLIRContext *context)
//       : OpConversionPattern<mlir::arith::ExtSIOp>(context) {}

//   using OpConversionPattern::OpConversionPattern;

//   LogicalResult matchAndRewrite(
//       mlir::arith::ExtSIOp op, OpAdaptor adaptor,
//       ConversionPatternRewriter &rewriter) const override {
//     ImplicitLocOpBuilder b(op.getLoc(), rewriter);

//     auto outType = convertArithToCGGIQuartType(
//         cast<IntegerType>(op.getResult().getType()), op->getContext());
//     auto castOp = b.create<cggi::CastOp>(op.getLoc(), outType,
//     adaptor.getIn());

//     rewriter.replaceOp(op, castOp);
//     return success();
//   }
// };

// struct ConvertShRUIOp : public OpConversionPattern<mlir::arith::ShRUIOp> {
//   ConvertShRUIOp(mlir::MLIRContext *context)
//       : OpConversionPattern<mlir::arith::ShRUIOp>(context) {}

//   using OpConversionPattern::OpConversionPattern;

//   LogicalResult matchAndRewrite(
//       mlir::arith::ShRUIOp op, OpAdaptor adaptor,
//       ConversionPatternRewriter &rewriter) const override {
//     ImplicitLocOpBuilder b(op.getLoc(), rewriter);

//     auto cteShiftSizeOp =
//     op.getRhs().getDefiningOp<mlir::arith::ConstantOp>();

//     if (cteShiftSizeOp) {
//       auto outputType = adaptor.getLhs().getType();

//       auto shiftAmount = cast<IntegerAttr>(cteShiftSizeOp.getValue())
//                              .getValue()
//                              .getSExtValue();

//       auto inputValue =
//           mlir::IntegerAttr::get(rewriter.getI8Type(), (int8_t)shiftAmount);
//       auto cteOp = rewriter.create<mlir::arith::ConstantOp>(
//           op.getLoc(), rewriter.getI8Type(), inputValue);

//       auto shiftOp =
//           b.create<cggi::ShiftRightOp>(outputType, adaptor.getLhs(), cteOp);
//       rewriter.replaceOp(op, shiftOp);

//       return success();
//     }

//     cteShiftSizeOp = op.getLhs().getDefiningOp<mlir::arith::ConstantOp>();

//     auto outputType = adaptor.getRhs().getType();

//     auto shiftAmount =
//         cast<IntegerAttr>(cteShiftSizeOp.getValue()).getValue().getSExtValue();

//     auto inputValue = mlir::IntegerAttr::get(rewriter.getI8Type(),
//     shiftAmount); auto cteOp = rewriter.create<mlir::arith::ConstantOp>(
//         op.getLoc(), rewriter.getI8Type(), inputValue);

//     auto shiftOp =
//         b.create<cggi::ShiftRightOp>(outputType, adaptor.getLhs(), cteOp);
//     rewriter.replaceOp(op, shiftOp);
//     rewriter.replaceOp(op.getLhs().getDefiningOp(), cteOp);

//     return success();
//   }
// };

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
    auto intAttr = IntegerAttr::get(rewriter.getI8Type(), maxIntWidth >> 1);

    auto elemType = convertArithToCGGIType(
        IntegerType::get(op->getContext(), maxIntWidth), op->getContext());
    auto realTy = convertArithToCGGIType(
        IntegerType::get(op->getContext(), maxIntWidth >> 1), op->getContext());

    auto constantOp = b.create<mlir::arith::ConstantOp>(intAttr);

    SmallVector<Value> carries;
    SmallVector<Value> outputs;

    for (int i = 0; i < splitLhs.size(); ++i) {
      auto lowSum = b.create<cggi::AddOp>(splitLhs[i], splitRhs[i]);
      auto outputLsb = b.create<cggi::CastOp>(op.getLoc(), realTy, lowSum);
      auto outputLsbHigh =
          b.create<cggi::CastOp>(op.getLoc(), elemType, outputLsb);

      // Now all the outputs are 16b elements, wants presentation of 4x8b
      if (i != splitLhs.size() - 1) {
        auto carry = b.create<cggi::ShiftRightOp>(elemType, lowSum, constantOp);
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

    // auto lowSum0 = b.create<mlir::arith::AddIOp>(lhsElem0, rhsElem0);
    // auto lowSum1 = b.create<mlir::arith::AddIOp>(lhsElem1, rhsElem1);
    // auto lowSum2 = b.create<mlir::arith::AddIOp>(lhsElem2, rhsElem2);
    // auto lowSum3 = b.create<mlir::arith::AddIOp>(lhsElem3, rhsElem3);

    // auto output0Lsb = b.create<mlir::arith::TruncIOp>(realTy, lowSum0);
    // auto output0LsbHigh = b.create<mlir::arith::ExtUIOp>(elemTy, output0Lsb);

    // auto output1Lsb = b.create<mlir::arith::TruncIOp>(realTy, lowSum1);
    // auto output1LsbHigh = b.create<mlir::arith::ExtUIOp>(elemTy, output1Lsb);

    // auto output2Lsb = b.create<mlir::arith::TruncIOp>(realTy, lowSum2);
    // auto output2LsbHigh = b.create<mlir::arith::ExtUIOp>(elemTy, output2Lsb);

    // auto output3Lsb = b.create<mlir::arith::TruncIOp>(realTy, lowSum3);
    // auto output3LsbHigh = b.create<mlir::arith::ExtUIOp>(elemTy, output3Lsb);

    // // Now all the outputs are 16b elements, wants presentation of 4x8b
    // auto carry0 =
    //     b.create<mlir::arith::ShRUIOp>(lowSum0, constantOp.getResult());
    // auto carry1 =
    //     b.create<mlir::arith::ShRUIOp>(lowSum1, constantOp.getResult());
    // auto carry2 =
    //     b.create<mlir::arith::ShRUIOp>(lowSum2, constantOp.getResult());

    // auto high1 = b.create<mlir::arith::AddIOp>(output1LsbHigh, carry0);
    // auto high2 = b.create<mlir::arith::AddIOp>(output2LsbHigh, carry1);
    // auto high3 = b.create<mlir::arith::AddIOp>(output3LsbHigh, carry2);

    // Value resultVec = constructResultTensor(
    //     rewriter, loc, newTy, ou);
    // rewriter.replaceOp(op, resultVec);
    // return success();
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
          // Allow use of constant if it is used to denote the size of a shift
          bool usedByShift = llvm::any_of(op->getUsers(), [&](Operation *user) {
            return isa<cggi::ShiftRightOp>(user);
          });
          return (isa<IndexType>(op.getValue().getType()) || (usedByShift));
        });

    patterns.add<
        ConvertQuartConstantOp, ConvertQuartExtUI, ConvertQuartAddI,
        // ConvertTruncIOp, ConvertExtUIOp,
        // ConvertShRUIOp,
        // ConvertExtSIOp,
        // ConvertBinOp<mlir::arith::AddIOp, cggi::AddOp>,
        // ConvertBinOp<mlir::arith::MulIOp, cggi::MulOp>,
        // ConvertBinOp<mlir::arith::SubIOp, cggi::SubOp>,
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
