#include "lib/Target/TfheRustHL/TfheRustHLEmitter.h"

#include <cassert>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <functional>
#include <iterator>
#include <numeric>
#include <string>
#include <string_view>
#include <tuple>
#include <utility>

#include "lib/Analysis/SelectVariableNames/SelectVariableNames.h"
#include "lib/Dialect/TfheRust/IR/TfheRustDialect.h"
#include "lib/Dialect/TfheRust/IR/TfheRustOps.h"
#include "lib/Dialect/TfheRust/IR/TfheRustTypes.h"
#include "lib/Target/TfheRust/Utils.h"
#include "lib/Target/TfheRustHL/TfheRustHLTemplates.h"
#include "lib/Transforms/MemrefToArith/Utils.h"
#include "lib/Utils/TargetUtils.h"
#include "llvm/include/llvm/ADT/TypeSwitch.h"          // from @llvm-project
#include "llvm/include/llvm/Support/FormatVariadic.h"  // from @llvm-project
#include "llvm/include/llvm/Support/raw_ostream.h"     // from @llvm-project
#include "mlir/include/mlir/Dialect/Affine/Analysis/AffineAnalysis.h"  // from @llvm-project
#include "mlir/include/mlir/Dialect/Affine/IR/AffineOps.h"  // from @llvm-project
#include "mlir/include/mlir/Dialect/Affine/IR/AffineValueMap.h"  // from @llvm-project
#include "mlir/include/mlir/Dialect/Affine/Utils.h"      // from @llvm-project
#include "mlir/include/mlir/Dialect/Arith/IR/Arith.h"    // from @llvm-projectx
#include "mlir/include/mlir/Dialect/Func/IR/FuncOps.h"   // from @llvm-project
#include "mlir/include/mlir/Dialect/MemRef/IR/MemRef.h"  // from @llvm-project
#include "mlir/include/mlir/Dialect/Tensor/IR/Tensor.h"  // from @llvm-project
#include "mlir/include/mlir/IR/BuiltinAttributes.h"      // from @llvm-project
#include "mlir/include/mlir/IR/BuiltinOps.h"             // from @llvm-project
#include "mlir/include/mlir/IR/BuiltinTypeInterfaces.h"  // from @llvm-project
#include "mlir/include/mlir/IR/BuiltinTypes.h"           // from @llvm-project
#include "mlir/include/mlir/IR/DialectRegistry.h"        // from @llvm-project
#include "mlir/include/mlir/IR/ImplicitLocOpBuilder.h"   // from @llvm-project
#include "mlir/include/mlir/IR/Types.h"                  // from @llvm-project
#include "mlir/include/mlir/IR/Value.h"                  // from @llvm-project
#include "mlir/include/mlir/IR/ValueRange.h"             // from @llvm-project
#include "mlir/include/mlir/IR/Visitors.h"               // from @llvm-project
#include "mlir/include/mlir/Support/LLVM.h"              // from @llvm-project
#include "mlir/include/mlir/Support/LogicalResult.h"     // from @llvm-project
#include "mlir/include/mlir/Tools/mlir-translate/Translation.h"  // from @llvm-project

#define DEBUG_TYPE "emit-tfhe-rust-hl-bool"

namespace mlir {
namespace heir {
namespace tfhe_rust {

namespace {

// getRustIntegerType returns the width of the closest builtin integer type.
FailureOr<int> getRustIntegerType(int width) {
  for (int candidate : {8, 16, 32, 64, 128}) {
    if (width <= candidate) {
      return candidate;
    }
  }
  return failure();
}

}  // namespace

void registerToTfheRustHLTranslation() {
  TranslateFromMLIRRegistration reg(
      "emit-tfhe-rust-hl", "translate the tfhe-rs dialect to HL Rust code",
      [](Operation *op, llvm::raw_ostream &output) {
        return translateToTfheRustHL(op, output, /*packedAPI=*/false);
      },
      [](DialectRegistry &registry) {
        registry.insert<func::FuncDialect, tfhe_rust::TfheRustDialect,
                        affine::AffineDialect, arith::ArithDialect,
                        tensor::TensorDialect, memref::MemRefDialect>();
      });
}

LogicalResult translateToTfheRustHL(Operation *op, llvm::raw_ostream &os,
                                    bool packedAPI) {
  SelectVariableNames variableNames(op);
  TfheRustHLEmitter emitter(os, &variableNames, packedAPI);
  LogicalResult result = emitter.translate(*op);
  return result;
}

LogicalResult TfheRustHLEmitter::translate(Operation &op) {
  LogicalResult status =
      llvm::TypeSwitch<Operation &, LogicalResult>(op)
          // Builtin ops
          .Case<ModuleOp>([&](auto op) { return printOperation(op); })
          // Func ops
          .Case<func::FuncOp, func::ReturnOp>(
              [&](auto op) { return printOperation(op); })
          // Affine ops
          .Case<affine::AffineForOp, affine::AffineYieldOp,
                affine::AffineLoadOp, affine::AffineStoreOp>(
              [&](auto op) { return printOperation(op); })
          // Arith ops
          .Case<arith::ConstantOp, arith::IndexCastOp, arith::ShRSIOp,
                arith::ShLIOp, arith::TruncIOp, arith::AndIOp>(
              [&](auto op) { return printOperation(op); })
          // MemRef ops
          .Case<memref::AllocOp, memref::LoadOp, memref::StoreOp>(
              [&](auto op) { return printOperation(op); })
          // TfheRust ops
          .Case<AddOp, MulOp, CreateTrivialOp>(
              [&](auto op) { return printOperation(op); })
          // Tensor ops
          .Case<tensor::ExtractOp, tensor::FromElementsOp>(
              [&](auto op) { return printOperation(op); })
          .Default([&](Operation &) {
            return op.emitOpError("unable to find printer for op");
          });

  if (failed(status)) {
    op.emitOpError(llvm::formatv("Failed to translate op {0}", op.getName()));
    return failure();
  }
  return success();
}

LogicalResult TfheRustHLEmitter::printOperation(ModuleOp moduleOp) {
  os << (packedAPI ? kFPGAModulePrelude : kModulePrelude) << "\n";
  for (Operation &op : moduleOp) {
    if (failed(translate(op))) {
      return failure();
    }
  }

  return success();
}

LogicalResult TfheRustHLEmitter::printOperation(func::FuncOp funcOp) {
  if (failed(tfhe_rust::canEmitFuncForTfheRust(funcOp))) {
    // Return success implies print nothing, and note the called function
    // emits a warning.
    return success();
  }

  os << "pub fn " << funcOp.getName() << "(\n";
  os.indent();
  for (Value arg : funcOp.getArguments()) {
    auto argName = variableNames->getNameForValue(arg);
    os << argName << ": &";
    if (failed(emitType(arg.getType()))) {
      return funcOp.emitOpError()
             << "Failed to emit tfhe-rs bool type " << arg.getType();
    }
    os << ",\n";
  }
  os.unindent();
  os << ") -> ";

  if (funcOp.getNumResults() == 1) {
    Type result = funcOp.getResultTypes()[0];
    if (failed(emitType(result))) {
      return funcOp.emitOpError()
             << "Failed to emit tfhe-rs bool type " << result;
    }
  } else {
    auto result = commaSeparatedTypes(
        funcOp.getResultTypes(), [&](Type type) -> FailureOr<std::string> {
          auto result = convertType(type);
          if (failed(result)) {
            return funcOp.emitOpError()
                   << "Failed to emit tfhe-rs bool type " << type;
          }
          return result;
        });
    os << "(" << result.value() << ")";
  }

  os << " {\n";
  os.indent();

  for (Block &block : funcOp.getBlocks()) {
    for (Operation &op : block.getOperations()) {
      if (failed(translate(op))) {
        return failure();
      }
    }
  }

  os.unindent();
  os << "}\n";
  return success();
}

LogicalResult TfheRustHLEmitter::printOperation(func::ReturnOp op) {
  std::function<std::string(Value)> valueOrClonedValue = [&](Value value) {
    auto *suffix = "";
    if (isa<BlockArgument>(value)) {
      suffix = ".clone()";
    }
    if (isa<tensor::FromElementsOp>(value.getDefiningOp())) {
      suffix = ".into_iter().cloned().collect()";
    }
    if (isa<memref::AllocOp>(value.getDefiningOp())) {
      // MemRefs (BTreeMap<(usize, ...), Ciphertext>) must be converted to
      // Vec<Ciphertext>
      suffix = ".into_values().collect()";
    };
    return variableNames->getNameForValue(value) + suffix;
  };

  if (op.getNumOperands() == 1) {
    os << valueOrClonedValue(op.getOperands()[0]) << "\n";
    return success();
  }

  os << "(" << commaSeparatedValues(op.getOperands(), valueOrClonedValue)
     << ")\n";
  return success();
}

void TfheRustHLEmitter::emitAssignPrefix(Value result) {
  os << "let " << variableNames->getNameForValue(result) << " = ";
}

void TfheRustHLEmitter::emitReferenceConversion(Value value) {
  // auto tensorType = dyn_cast<TensorType>(value.getType());

  // if (isa<EncryptedBoolType>(tensorType.getElementType())) {
  //   auto varName = variableNames->getNameForValue(value);

  auto varName = variableNames->getNameForValue(value);

  os << "let " << varName << "_ref = " << varName << ".clone();\n";
  os << "let " << varName << "_ref: Vec<&Ciphertext> = " << varName
     << ".iter().collect();\n";
}

LogicalResult TfheRustHLEmitter::printSksMethod(
    ::mlir::Value result, ::mlir::Value sks, ::mlir::ValueRange nonSksOperands,
    std::string_view op, SmallVector<std::string> operandTypes) {
  // If using the packed API, then emit single boolean operations as a packed
  // operations with a single gate
  std::string gateStr = StringRef(op).upper();
  auto *opParent = nonSksOperands[0].getDefiningOp();

  size_t numberOfOperands = 0;

  // Handle element-wise boolean gate operations with tensor type operands and
  // results.
  if (isa<TensorType>(result.getType())) {
    auto resType = mlir::dyn_cast<TensorType>(result.getType());
    numberOfOperands = resType.getNumElements();
  } else {
    numberOfOperands = 1;
  }

  if (!opParent) {
    for (auto nonSksOperand : nonSksOperands) {
      emitReferenceConversion(nonSksOperand);
    }
  }

  emitAssignPrefix(result);
  os << variableNames->getNameForValue(sks);

  // parse the not gate
  if (!gateStr.compare("NOT")) {
    os << ".packed_not(\n";
  } else {
    os << ".packed_gates( \n &vec![";

    for (size_t i = 0; i < numberOfOperands; i++) {
      os << "Gate::" << gateStr << ", ";
    }

    os << "],\n";
  }

  os << commaSeparatedValues(
      nonSksOperands, [&, numberOfOperands](Value value) {
        std::string prefix;
        std::string suffix;

        tie(prefix, suffix) = checkOrigin(value);
        if (numberOfOperands == 1) {
          prefix = "&vec![&";
          suffix = "]" + suffix;
        }

        return prefix + variableNames->getNameForValue(value) + suffix;
      });

  if (numberOfOperands == 1) {
    os << ")[0].clone();\n";
  } else {
    os << ");\n";
  }

  // Check that this translation can only be used by non-tensor operands
  if (!isa<TensorType>(nonSksOperands[0].getType())) {
    emitAssignPrefix(result);

    auto *operandTypesIt = operandTypes.begin();
    os << variableNames->getNameForValue(sks) << "." << op << "(";
    os << commaSeparatedValues(nonSksOperands, [&](Value value) {
      auto *prefix = value.getType().hasTrait<PassByReference>() ? "&" : "";
      // First check if a DefiningOp exists
      // if not: comes from function definition
      mlir::Operation *op = value.getDefiningOp();
      if (op) {
        auto referencePredicate =
            isa<tensor::ExtractOp>(op) || isa<memref::LoadOp>(op);
        prefix = referencePredicate ? "" : prefix;
      } else {
        prefix = "";
      }

      return prefix + variableNames->getNameForValue(value) +
             (!operandTypes.empty() ? " as " + *operandTypesIt++ : "");
    });

    os << ");\n";

    return success();
  }

  return failure();
}

LogicalResult TfheRustHLEmitter::printOperation(CreateTrivialOp op) {
  return printSksMethod(op.getResult(), op.getServerKey(), {op.getValue()},
                        "create_trivial", {"u64"});
}

LogicalResult TfheRustHLEmitter::printOperation(affine::AffineForOp op) {
  if (op.getStepAsInt() > 1) {
    return op.emitOpError() << "AffineForOp has step > 1";
  }
  os << "for " << variableNames->getNameForValue(op.getInductionVar()) << " in "
     << op.getConstantLowerBound() << ".." << op.getConstantUpperBound()
     << " {\n";
  os.indent();

  auto res = op.getBody()->walk([&](Operation *op) {
    if (failed(translate(*op))) {
      return WalkResult::interrupt();
    }
    return WalkResult::advance();
  });

  if (res.wasInterrupted()) {
    return failure();
  }

  os.unindent();
  os << "}\n";
  return success();
}

LogicalResult TfheRustHLEmitter::printOperation(affine::AffineYieldOp op) {
  if (op->getNumResults() != 0) {
    return op.emitOpError() << "AffineYieldOp has non-zero number of results";
  }
  return success();
}

LogicalResult TfheRustHLEmitter::printOperation(arith::ConstantOp op) {
  auto valueAttr = op.getValue();
  if (isa<IntegerType>(op.getType()) &&
      op.getType().getIntOrFloatBitWidth() == 1) {
    os << "let " << variableNames->getNameForValue(op.getResult())
       << " : bool = ";
    os << (cast<IntegerAttr>(valueAttr).getValue().isZero() ? "false" : "true")
       << ";\n";
    return success();
  }

  emitAssignPrefix(op.getResult());
  if (auto intAttr = dyn_cast<IntegerAttr>(valueAttr)) {
    os << intAttr.getValue() << ";\n";
  } else {
    return op.emitError() << "Unknown constant type " << valueAttr.getType();
  }
  return success();
}

LogicalResult TfheRustHLEmitter::printOperation(arith::IndexCastOp op) {
  emitAssignPrefix(op.getOut());
  os << variableNames->getNameForValue(op.getIn()) << " as ";
  if (failed(emitType(op.getOut().getType()))) {
    return op.emitOpError()
           << "Failed to emit index cast type " << op.getOut().getType();
  }
  os << ";\n";
  return success();
}

LogicalResult TfheRustHLEmitter::printBinaryOp(::mlir::Value result,
                                               ::mlir::Value lhs,
                                               ::mlir::Value rhs,
                                               std::string_view op) {
  emitAssignPrefix(result);
  os << variableNames->getNameForValue(lhs) << " " << op << " "
     << variableNames->getNameForValue(rhs) << ";\n";
  return success();
}

LogicalResult TfheRustHLEmitter::printOperation(::mlir::arith::ShLIOp op) {
  return printBinaryOp(op.getResult(), op.getLhs(), op.getRhs(), "<<");
}

LogicalResult TfheRustHLEmitter::printOperation(::mlir::arith::AndIOp op) {
  return printBinaryOp(op.getResult(), op.getLhs(), op.getRhs(), "&");
}

LogicalResult TfheRustHLEmitter::printOperation(::mlir::arith::ShRSIOp op) {
  return printBinaryOp(op.getResult(), op.getLhs(), op.getRhs(), ">>");
}

LogicalResult TfheRustHLEmitter::printOperation(::mlir::arith::TruncIOp op) {
  emitAssignPrefix(op.getResult());
  os << variableNames->getNameForValue(op.getIn());
  if (isa<IntegerType>(op.getType()) &&
      op.getType().getIntOrFloatBitWidth() == 1) {
    // Compare with zero to truncate to a boolean.
    os << " != 0";
  } else {
    os << " as ";
    if (failed(emitType(op.getType()))) {
      return op.emitOpError()
             << "Failed to emit truncated type " << op.getType();
    }
  }
  os << ";\n";
  return success();
}

// Use a BTreeMap<(usize, ...), Ciphertext>.
LogicalResult TfheRustHLEmitter::printOperation(memref::AllocOp op) {
  os << "let mut " << variableNames->getNameForValue(op.getMemref())
     << " : BTreeMap<("
     << std::accumulate(
            std::next(op.getMemref().getType().getShape().begin()),
            op.getMemref().getType().getShape().end(), std::string("usize"),
            [&](const std::string &a, int64_t value) { return a + ", usize"; })
     << "), ";
  if (failed(emitType(op.getMemref().getType().getElementType()))) {
    return op.emitOpError() << "Failed to get memref element type";
  }

  os << "> = BTreeMap::new();\n";
  return success();
}

// Store into a BTreeMap<(usize, ...), Ciphertext>
LogicalResult TfheRustHLEmitter::printOperation(memref::StoreOp op) {
  // We assume here that the indices are SSA values (not integer attributes).
  os << variableNames->getNameForValue(op.getMemref());
  os << ".insert((" << commaSeparatedValues(op.getIndices(), [&](Value value) {
    return variableNames->getNameForValue(value) + std::string(" as usize");
  }) << "), ";

  // Note: we may not need to clone all the time, but the BTreeMap stores
  // Ciphertexts, not &Ciphertexts. This is because results computed inside for
  // loops will not live long enough.
  const auto *suffix = ".clone()";
  os << variableNames->getNameForValue(op.getValueToStore()) << suffix
     << ");\n";
  return success();
}

// Produces a &Ciphertext
LogicalResult TfheRustHLEmitter::printOperation(memref::LoadOp op) {
  // We assume here that the indices are SSA values (not integer attributes).
  if (isa<BlockArgument>(op.getMemref())) {
    emitAssignPrefix(op.getResult());
    os << "&" << variableNames->getNameForValue(op.getMemRef()) << "["
       << flattenIndexExpression(op.getMemRefType(), op.getIndices(),
                                 [&](Value value) {
                                   return variableNames->getNameForValue(value);
                                 })
       << "];\n";
    return success();
  }

  // Treat this as a BTreeMap
  emitAssignPrefix(op.getResult());
  os << "&" << variableNames->getNameForValue(op.getMemref()) << ".get(&("
     << commaSeparatedValues(op.getIndices(),
                             [&](Value value) {
                               return variableNames->getNameForValue(value) +
                                      " as usize";
                             })
     << ")).unwrap();\n";
  return success();
}

// FIXME?: This is a hack to get the index of the value
static int extractIntFromValue(Value value) {
  auto ctOp = dyn_cast<arith::ConstantOp>(value.getDefiningOp());
  return cast<IntegerAttr>(ctOp.getValue()).getValue().getSExtValue();
}

// Store into a BTreeMap<(usize, ...), Ciphertext>
LogicalResult TfheRustHLEmitter::printOperation(affine::AffineStoreOp op) {
  // We assume here that the indices are SSA values (not integer attributes).

  OpBuilder builder(op->getContext());
  auto indices = affine::expandAffineMap(builder, op->getLoc(), op.getMap(),
                                         op.getIndices());

  os << variableNames->getNameForValue(op.getMemref());
  os << ".insert((" << commaSeparatedValues(indices.value(), [&](Value value) {
    return std::to_string(extractIntFromValue(value)) +
           std::string(" as usize");
  }) << "), ";

  // Note: we may not need to clone all the time, but the BTreeMap stores
  // Ciphertexts, not &Ciphertexts. This is because results computed inside for
  // loops will not live long enough.

  const auto *suffix = ".clone()";
  os << variableNames->getNameForValue(op.getValueToStore()) << suffix
     << ");\n";
  return success();
}

// Produces a &Ciphertext
LogicalResult TfheRustHLEmitter::printOperation(affine::AffineLoadOp op) {
  OpBuilder builder(op->getContext());
  auto indices = affine::expandAffineMap(builder, op->getLoc(), op.getMap(),
                                         op.getIndices());

  if (isa<BlockArgument>(op.getMemref())) {
    emitAssignPrefix(op.getResult());

    os << "&" << variableNames->getNameForValue(op.getMemRef()) << "["
       << flattenIndexExpression(
              op.getMemRefType(), indices.value(),
              [&](Value value) {
                return std::to_string(extractIntFromValue(value));
              })
       << "];\n";
    return success();
  }

  // Treat this as a BTreeMap
  emitAssignPrefix(op.getResult());
  os << "&" << variableNames->getNameForValue(op.getMemref()) << ".get(&("
     << commaSeparatedValues(
            indices.value(),
            [&](Value value) {
              return std::to_string(extractIntFromValue(value));
            })
     << ")).unwrap();\n";
  return success();
}

// Produces a &Ciphertext
LogicalResult TfheRustHLEmitter::printOperation(tensor::ExtractOp op) {
  // We assume here that the indices are SSA values (not integer attributes).
  emitAssignPrefix(op.getResult());
  os << "&" << variableNames->getNameForValue(op.getTensor()) << "["
     << commaSeparatedValues(
            op.getIndices(),
            [&](Value value) { return variableNames->getNameForValue(value); })
     << "];\n";
  return success();
}

// Need to produce a Vec<&Ciphertext>
LogicalResult TfheRustHLEmitter::printOperation(tensor::FromElementsOp op) {
  emitAssignPrefix(op.getResult());
  os << "vec![" << commaSeparatedValues(op.getOperands(), [&](Value value) {
    // Check if block argument, if so, clone.
    const auto *cloneStr = isa<BlockArgument>(value) ? ".clone()" : "";
    // Get the name of defining operation its dialect
    auto tfheOp =
        value.getDefiningOp()->getDialect()->getNamespace() == "tfhe_rust_bool";
    const auto *prefix = tfheOp ? "&" : "";
    return std::string(prefix) + variableNames->getNameForValue(value) +
           cloneStr;
  }) << "];\n";
  return success();
}

LogicalResult TfheRustHLEmitter::printOperation(BitAndOp op) {
  return printBinaryOp(op.getResult(), op.getLhs(), op.getRhs(), "&&");
}

LogicalResult TfheRustHLEmitter::printOperation(AddOp op) {
  return printBinaryOp(op.getResult(), op.getLhs(), op.getRhs(), "+");
}

LogicalResult TfheRustHLEmitter::printOperation(MulOp op) {
  return printBinaryOp(op.getResult(), op.getLhs(), op.getRhs(), "*");
}

FailureOr<std::string> TfheRustHLEmitter::convertType(Type type) {
  // Note: these are probably not the right type names to use exactly, and
  // they will need to chance to the right values once we try to compile it
  // against a specific API version.

  if (type.hasTrait<EncryptedInteger>()) {
    return std::string("Ciphertext");
  }

  return llvm::TypeSwitch<Type &, FailureOr<std::string>>(type)
      .Case<RankedTensorType>(
          [&](RankedTensorType type) -> FailureOr<std::string> {
            // Tensor types are emitted as vectors
            auto elementTy = convertType(type.getElementType());
            if (failed(elementTy)) return failure();
            return std::string("Vec<" + elementTy.value() + ">");
          })
      .Case<MemRefType>([&](MemRefType type) -> FailureOr<std::string> {
        // MemRef types are emitted as arrays
        auto elementTy = convertType(type.getElementType());
        if (failed(elementTy)) return failure();
        std::string res = elementTy.value();
        for (unsigned dim : llvm::reverse(type.getShape())) {
          res = llvm::formatv("[{0}; {1}]", res, dim);
        }
        return res;
      })
      .Case<IntegerType>([&](IntegerType type) -> FailureOr<std::string> {
        if (type.getWidth() == 1) {
          return std::string("bool");
        }
        auto width = getRustIntegerType(type.getWidth());
        if (failed(width)) return failure();
        return (type.isUnsigned() ? std::string("u") : "") + "i" +
               std::to_string(width.value());
      })
      .Case<ServerKeyType>([&](auto type) { return std::string("ServerKey"); })
      .Case<LookupTableType>(
          [&](auto type) { return std::string("LookupTableOwned"); })
      .Default([&](Type &) { return failure(); });
}

LogicalResult TfheRustHLEmitter::emitType(Type type) {
  auto result = convertType(type);
  if (failed(result)) {
    return failure();
  }
  os << result;
  return success();
}

std::pair<std::string, std::string> TfheRustHLEmitter::checkOrigin(
    Value value) {
  std::string prefix = "&";
  std::string suffix = "";
  // First check if a DefiningOp exists
  // if not: comes from function definition
  mlir::Operation *opParent = value.getDefiningOp();
  if (opParent) {
    if (!isa<tensor::FromElementsOp>(opParent) &&
        !isa<tensor::ExtractOp>(opParent))
      prefix = "";

  } else {
    prefix = "&";
    suffix = "_ref";
  }

  return std::make_pair(prefix, suffix);
}

TfheRustHLEmitter::TfheRustHLEmitter(raw_ostream &os,
                                     SelectVariableNames *variableNames,
                                     bool packedAPI)
    : os(os), variableNames(variableNames), packedAPI(packedAPI) {}
}  // namespace tfhe_rust
}  // namespace heir
}  // namespace mlir