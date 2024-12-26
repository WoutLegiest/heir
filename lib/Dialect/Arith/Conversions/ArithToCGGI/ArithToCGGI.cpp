#include "lib/Dialect/Arith/Conversions/ArithToCGGI/ArithToCGGI.h"

#include "lib/Dialect/CGGI/IR/CGGIDialect.h"
#include "lib/Dialect/CGGI/IR/CGGIOps.h"
#include "lib/Utils/ConversionUtils.h"
#include "mlir/include/mlir/Dialect/Arith/IR/Arith.h"  // from @llvm-project
#include "mlir/include/mlir/Transforms/DialectConversion.h"  // from @llvm-project

namespace mlir::heir::arith {

#define GEN_PASS_DEF_ARITHTOCGGI
#include "lib/Dialect/Arith/Conversions/ArithToCGGI/ArithToCGGI.h.inc"

// Remove this class if no type conversions are necessary
class ArithToCGGITypeConverter : public TypeConverter {
 public:
  ArithToCGGITypeConverter(MLIRContext *ctx) {
    addConversion([](Type type) { return type; });
    // FIXME: implement, replace FooType with the type that needs
    // to be converted or remove this class
    // addConversion([ctx](FooType type) -> Type {
    //   return type;
    // });
  }
};

// FIXME: rename to Convert<OpName>Op
// struct ConvertFooOp : public OpConversionPattern<FooOp> {
//   ConvertFooOp(mlir::MLIRContext *context)
//       : OpConversionPattern<FooOp>(context) {}

//   using OpConversionPattern::OpConversionPattern;

//   LogicalResult matchAndRewrite(
//       FooOp op, OpAdaptor adaptor,
//       ConversionPatternRewriter &rewriter) const override {
//     // FIXME: implement
//     return failure();
//   }
// };

struct ArithToCGGI : public impl::ArithToCGGIBase<ArithToCGGI> {
  void runOnOperation() override {
    MLIRContext *context = &getContext();
    auto *module = getOperation();
    ArithToCGGITypeConverter typeConverter(context);

    RewritePatternSet patterns(context);
    ConversionTarget target(*context);
    target.addLegalDialect<cggi::CGGIDialect>();
    target.addIllegalDialect<mlir::arith::ArithDialect>();

    patterns.add<ConvertBinOp<mlir::arith::AddIOp, cggi::AddOp>,
                 ConvertBinOp<mlir::arith::MulIOp, cggi::MulOp> >(typeConverter,
                                                                  context);

    addStructuralConversionPatterns(typeConverter, patterns, target);

    if (failed(applyPartialConversion(module, target, std::move(patterns)))) {
      return signalPassFailure();
    }
  }
};

}  // namespace mlir::heir::arith
